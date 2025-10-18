import re
import random
import json
import base64
import io
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

import os

from flask import render_template

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import request
from flask_login import current_user
from io import BytesIO
from functools import wraps
from flask import Response, make_response

# Import database collections from the main __init__.py
from . import collection, leads_collection

from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

# --- FIX STARTS HERE ---

# 1. Import the lazy-loading function itself, and any other functions you need
from backend.ml_model.classifier import _load_resources, _resources, categories

# Trigger loading and access the resources directly
_load_resources()
PRODUCT_CODE_TO_NAME_MAP = _resources["PRODUCT_CODE_TO_NAME_MAP"]
PRODUCT_HIERARCHY = _resources["PRODUCT_HIERARCHY"]

import os
from cryptography.fernet import Fernet

# Load the encryption key from the environment
# The application will crash if this key is not set, which is good for security.
FERNET_KEY = os.environ.get('FERNET_KEY')
if not FERNET_KEY:
    raise RuntimeError("FERNET_KEY is not set in the environment variables.")

cipher_suite = Fernet(FERNET_KEY.encode())

def encrypt_token(token):
    """Encrypts a token string."""
    if not token:
        return None
    return cipher_suite.encrypt(token.encode())

def decrypt_token(encrypted_token):
    """Decrypts an encrypted token string."""
    if not encrypted_token:
        return None
    return cipher_suite.decrypt(encrypted_token).decode()

def nocache(view):
    """
    A decorator to add HTTP headers that prevent browser caching.
    Useful for pages with sensitive data or forms to ensure freshness.
    """
    @wraps(view)
    def no_cache(*args, **kwargs):
        # Call the original view function to get the response
        response = view(*args, **kwargs)
        
        # If the view didn't return a full Response object, create one
        if not isinstance(response, Response):
            response = make_response(response)
        
        # Add headers to the response to disable caching
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        
        return response
    return no_cache



def convert_numpy_types(doc):
    if isinstance(doc, dict):
        return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in doc.items()}
    return doc



def extract_details(email):
    details = {
        "REF_NUMBER": "N/A",
        "APPROVER_NAME": "Mr. Sharma (Sales Head)",
        "COURIER_NAME": "[Courier Name]",  # Placeholder for sales team
        "TRACKING_ID": "[Tracking ID]",    # Placeholder for sales team
        "DELIVERY_DATE": "[Expected Timeframe]",  # Placeholder for sales team
        "URGENT": "URGENT: " if any(word in email.lower() for word in ["urgent", "immediate", "asap"]) else ""
    }
    ref_match = re.search(r"(order|ref|case)\s*(#|no\.?)?\s*(\d+)", email, re.IGNORECASE)
    if ref_match:
        details["REF_NUMBER"] = ref_match.group(3)
    return details



def get_my_reports_data():
    """
    Fetches and calculates personalized performance data for the logged-in salesperson.
    """
    sales_person_name = current_user.name
    
    # --- Lead Funnel Calculation (already correct) ---
    pipeline = [
        { '$match': { 'assigned_to': sales_person_name } },
        { '$group': { '_id': '$status', 'count': { '$sum': 1 } } }
    ]
    status_counts = list(leads_collection.aggregate(pipeline))
    my_lead_funnel = {item['_id']: item['count'] for item in status_counts}
    for status in ['new', 'contacted', 'interested', 'negative']:
        my_lead_funnel.setdefault(status, 0)
        
    # --- Personal SLA Trend Calculation (already correct) ---
    my_logs = list(collection.find({'resolved_by': sales_person_name}))
    my_logs_df = pd.DataFrame(my_logs)
    my_sla_trend_chart = create_sla_trend_chart(my_logs_df)

    # --- START: NEW PERSONAL PERFORMANCE METRICS ---
    my_performance = {}
    if not my_logs_df.empty:
        my_performance = {
            "name": sales_person_name,
            "emails_handled": len(my_logs_df),
            "avg_response_time": round(my_logs_df['delay_in_sec'].mean() / 3600, 1) if not my_logs_df['delay_in_sec'].empty else 0,
            "human_sla_compliance": my_logs_df['SLA_Met'].mean() * 100 if not my_logs_df['SLA_Met'].empty else 0,
            "satisfaction": round(random.uniform(3.8, 4.9), 1) # Placeholder for real survey data
        }
    # --- END: NEW PERSONAL PERFORMANCE METRICS ---

    context = {
        "my_sla_trend_chart": my_sla_trend_chart,
        "my_lead_funnel": my_lead_funnel,
        "my_performance": my_performance, # Add the new data to the context
    }
    return context




def humanize_product_tag(code: str) -> str:
    """Convert official product codes to human-readable names using the global map."""
    # Use .get() for safety. If a code is not found, it returns the code itself.
    return PRODUCT_CODE_TO_NAME_MAP.get(code, code)



def get_product_list_for_dropdown():
    """
    Converts the global PRODUCT_CODE_TO_NAME_MAP into a sorted list of
    dictionaries suitable for an HTML dropdown.
    """
    product_list = [{'code': code, 'display': name} for code, name in PRODUCT_CODE_TO_NAME_MAP.items()]
    # Sort the final list alphabetically by the display name for a clean dropdown
    return sorted(product_list, key=lambda x: x['display'])
    

def generate_product_insights(records):
    """
    Generate product insights using official product codes.
    The keys in product_counts will now be the official codes.
    """
    product_counts = defaultdict(int)
    category_capacity_map = defaultdict(lambda: defaultdict(int))

    for rec in records:
        # The 'products_detected' field now contains official codes
        codes = rec.get("products_detected", [])
        cat = rec.get("category", "unknown")
        for code in codes:
            product_counts[code] += 1
            # Extract capacity/size from the end of the code if possible
            # e.g., FEXT-CO2_PORTABLE-4.5KG -> 4.5KG
            try:
                capacity = code.rsplit('-', 1)[-1]
                category_capacity_map[cat][capacity] += 1
            except IndexError:
                pass  # Code doesn't have a capacity suffix

    # Format insights for display
    insights = []
    if records:
        # Note: 'total_records' is no longer the best denominator.
        # We'll use the total number of product tags detected for a more accurate percentage.
        total_tags_detected = sum(product_counts.values())
        if total_tags_detected > 0:
            # Sort by count, highest first, and take top 5
            sorted_items = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for code, count in sorted_items:
                readable_name = humanize_product_tag(code) # Uses our new, simple function
                insights.append({
                    'count': count,
                    'name': readable_name,
                    'raw_tag': code, # Store the official code
                    'percentage': round((count / total_tags_detected) * 100, 1)
                })

    return insights, product_counts, category_capacity_map

# ADD THIS NEW FUNCTION AT THE END OF THE FILE

def generate_pdf_from_template(template_name, data, css_string=None):
    """
    Generates a PDF from a given HTML template and data using WeasyPrint.

    Args:
        template_name (str): The filename of the HTML template in the 'templates' folder.
        data (dict): A dictionary of data to be passed to the HTML template.
        css_string (str, optional): A string containing custom CSS for the PDF.

    Returns:
        tuple: A tuple containing the PDF file in a BytesIO buffer and a generated filename.
    """
    try:
        # Render the HTML template with the provided data
        html_content = render_template(template_name, **data)

        # Use a default CSS if none is provided
        if not css_string:
            css_string = """
            @page { size: A4; margin: 1in; }
            body { font-family: Arial, sans-serif; line-height: 1.4; color: #333; }
            h1, h2, h3 { color: #0056b3; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .header, .footer { text-align: center; }
            """
        
        # FIXED: Proper WeasyPrint initialization
        font_config = FontConfiguration()
        
        # Create HTML object with keyword argument
        html_doc = HTML(string=html_content)
        
        # Create CSS object with keyword argument
        css_doc = CSS(string=css_string, font_config=font_config)
        
        # Write the PDF to an in-memory buffer
        pdf_buffer = io.BytesIO()
        html_doc.write_pdf(pdf_buffer, stylesheets=[css_doc], font_config=font_config)
        pdf_buffer.seek(0)
        
        # Create a dynamic filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}.pdf"
        
        return pdf_buffer, filename
        
    except Exception as e:
        # In a real app, you would log this error.
        print(f"Error in PDF generation: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"PDF generation failed: {str(e)}")

def create_performance_leaderboard(team_data):
    """
    Generate a smooth vertical bar chart like the 'Chartly' reference with pastel theme.
    """
    try:
        if not team_data:
            return ""

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_alpha(0.0)  # Transparent background
        ax.set_facecolor('#FFFFFF00')

        df = pd.DataFrame(team_data)
        df['score'] = pd.to_numeric(df.get('score'), errors='coerce').fillna(0)
        df = df.sort_values('score', ascending=False).head(5)

        names = df['name'].tolist()
        scores = df['score'].tolist()

        # Use a soft purple color like the reference
        base_color = '#A06FF5'
        edge_color = '#7B4FEF'

        bars = ax.bar(
            names,
            scores,
            color=base_color,
            edgecolor=edge_color,
            linewidth=1,
            width=0.5
        )

        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#333333'
            )

        # Style adjustments
        ax.set_title('Team Response Score', fontsize=13, color='#555', pad=15)
        ax.set_ylim(0, max(scores) + 15)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.tick_params(axis='x', colors='#444')
        ax.tick_params(axis='y', length=0, colors='#aaa')

        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Error generating leaderboard chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def create_sla_summary_chart(context):
    """Generate SLA compliance visualization"""
    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(['SLA Met', 'Breached'], 
               [context['sla_met_pct'], 100 - context['sla_met_pct']],
               color=['#4CAF50', '#F44336'])
        ax.set_title('SLA Compliance Overview', pad=10)
        ax.set_ylim(0, 100)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating SLA chart: {str(e)}")
        return ""


def create_product_demand_chart(logs): # Changed context to logs for clarity
    """Generate top products visualization with thicker bars and count labels."""
    try:
        # --- 1. Data Preparation ---
        if not logs:
            return ""

        product_counts = defaultdict(int)
        for rec in logs:
            for code in rec.get("products_detected", []):
                product_counts[code] += 1
        
        if not product_counts:
            return ""

        # Get top 5 products
        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Reverse the list for top-to-bottom display on the chart
        top_products.reverse()

        # Prepare data for plotting
        # THE FIX: Add the count to the product name for a richer label
        names = [f"{humanize_product_tag(p[0])} ({p[1]})" for p in top_products]
        counts = [p[1] for p in top_products]
        
        # --- 2. Chart Creation and Styling ---
        fig, ax = plt.subplots(figsize=(10, 6)) # Slightly larger figure for better spacing
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('#FFFFFF00')

        # THE FIX: Increase bar thickness using the 'height' parameter
        bars = ax.barh(names, counts, color='#0D6EFD', height=0.6, align='center')
        
        # Clean up axes and remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False) # Hide x-axis numbers as the label now has the count

        # Customize tick labels (the product names)
        ax.tick_params(axis='y', length=0, labelsize=11, labelcolor='gray')
        
        # Set title
        ax.set_title('Top Requested Products', color='gray', fontsize=14, pad=20)
        
        plt.tight_layout(pad=1.5)

        # --- 3. Saving to Buffer ---
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode()

    except Exception as e:
        print(f"Error generating product demand chart: {str(e)}")
        return ""
    


def create_sla_trend_chart(df):
    """
    Generate a MODERN and ROBUST SLA compliance trend chart.
    This version correctly handles an empty DataFrame and processes raw log data.
    """
    try:
        # THE FIX: Check for an empty DataFrame using .empty
        # Also check that the required columns exist before proceeding.
        if df.empty or 'timestamp' not in df.columns or 'SLA_Met' not in df.columns:
            return ""

        # This part correctly processes the raw data into a plottable format
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        sla_trends = df.groupby("date")["SLA_Met"].mean().reset_index()
        sla_trends["SLA_Met"] *= 100

        # A trend line needs at least two points to be meaningful
        if len(sla_trends) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("#00000000")

        line_color = '#0D6EFD'
        dark_text_color = "#495057" 
        light_grid_color = (0.0, 0.0, 0.0, 0.1) 
        
        # Use the processed 'sla_trends' DataFrame for plotting
        ax.plot(sla_trends["date"], sla_trends["SLA_Met"], 
                color=line_color, linewidth=2.5, marker='o', markersize=8, 
                markerfacecolor='white', markeredgecolor=line_color,
                markeredgewidth=2.5, zorder=3)
        
        ax.fill_between(
            sla_trends["date"], sla_trends["SLA_Met"], 
            color=line_color, alpha=0.1, zorder=2)

        ax.set_ylim(0, 105)
        ax.set_ylabel("")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color(light_grid_color)

        ax.tick_params(axis='x', colors=dark_text_color, length=0, pad=10, labelsize=11)
        ax.tick_params(axis='y', colors=dark_text_color, length=0, pad=10, labelsize=11)
        
        ax.grid(axis='y', linestyle='--', alpha=0.5, color=light_grid_color, zorder=1)
        
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, transparent=True)
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Error generating SLA trend chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
    

def create_automation_metrics_chart(metrics):
    """Visualize AI-human collaboration metrics"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Human intervention metrics
        ax1.barh(
            ['Escalated Cases', 'Human Response', 'QA Review'], 
            [metrics['escalation_rate'], metrics['human_response_time'], metrics['qa_time']],
            color=['#ff7f0e', '#1f77b4', '#2ca02c']
        )
        ax1.set_title('Human Intervention Metrics')
        ax1.set_xlabel('Time (hours) / Percentage')
        
        # System performance
        ax2.bar(
            ['Auto-Reply Success', 'Human Correction', 'Confidence'],
            [metrics['success_rate'], metrics['correction_rate'], metrics['avg_confidence']],
            color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
        ax2.set_title('System Performance')
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating metrics chart: {str(e)}")
        return ""

# =============================================
# END OF CHART FUNCTIONS
# =============================================

def calculate_product_trends(current_logs, previous_logs):
    """Calculates the percentage change in product mentions between two periods."""
    
    # Get product counts for the current period
    _, current_counts, _ = generate_product_insights(current_logs)
    
    # Get product counts for the previous period
    _, previous_counts, _ = generate_product_insights(previous_logs)

    all_product_codes = set(current_counts.keys()) | set(previous_counts.keys())
    
    trends = []
    for code in all_product_codes:
        current_val = current_counts.get(code, 0)
        prev_val = previous_counts.get(code, 0)
        
        if prev_val > 0:
            # Standard percentage change calculation
            change = ((current_val - prev_val) / prev_val) * 100
        elif current_val > 0:
            # If a product is new, its trend is effectively infinite. We'll show a large positive number.
            change = 200.0 # Represents a "new" product trend
        else:
            # If a product appeared in neither, there's no trend
            continue

        trends.append({
            "name": humanize_product_tag(code),
            "change": round(change)
        })

    # Separate into "up" and "down" trends and sort them
    trending_up = sorted([p for p in trends if p['change'] > 0], key=lambda x: x['change'], reverse=True)[:3]
    trending_down = sorted([p for p in trends if p['change'] < 0], key=lambda x: x['change'])[:3]
    
    return {"up": trending_up, "down": trending_down}



# FILE: backend/app/utils.py

# Replace the old get_dashboard_data with this new version
def get_dashboard_data(selected_category="all", sla_filter="all", product_filter="all", date_from_str="", date_to_str=""):
    """
    Centralized function to fetch and process dashboard data.
    This version accepts filter parameters as arguments for better testability and reliability.
    """
    # --- 1. Sanitize inputs to handle Undefined objects and None values ---
    from jinja2 import Undefined
    
    selected_category = str(selected_category) if selected_category and not isinstance(selected_category, Undefined) else "all"
    sla_filter = str(sla_filter) if sla_filter and not isinstance(sla_filter, Undefined) else "all"
    product_filter = str(product_filter) if product_filter and not isinstance(product_filter, Undefined) else "all"
    date_from_str = str(date_from_str) if date_from_str and not isinstance(date_from_str, Undefined) else ""
    date_to_str = str(date_to_str) if date_to_str and not isinstance(date_to_str, Undefined) else ""
    

    # --- 2. Build the MongoDB query from filters ---
    query = {}
    if selected_category != "all":
        query["category"] = selected_category
    if sla_filter == "met":
        query["SLA_Met"] = True
    elif sla_filter == "breached":
        query["SLA_Met"] = False
    if product_filter != "all":
        query["products_detected"] = product_filter
    
    date_filter = {}
    if date_from_str:
        try: date_filter["$gte"] = datetime.strptime(date_from_str, "%Y-%m-%d")
        except ValueError: pass
    if date_to_str:
        try: date_filter["$lt"] = datetime.strptime(date_to_str, "%Y-%m-%d") + timedelta(days=1)
        except ValueError: pass
    if date_filter:
        query["timestamp"] = date_filter

    # --- 3. Fetch data ---
    logs = list(collection.find(query))
    all_logs = list(collection.find()) # For overall trends and filter options
    
    # ... THE REST OF YOUR FUNCTION REMAINS EXACTLY THE SAME ...
    # From here down, you don't need to change anything in this function.
    # Just make sure the function signature and the query-building part above are updated.

    # --- 4. Handle all "No Data" Scenarios ---
    if not logs:
        # Prepare a complete but zeroed-out context
        empty_context = {
            "no_data_message": "No data available for the selected filters.",
            "all_categories": [],
            "all_products": get_product_list_for_dropdown(),
            "selected_category": selected_category, "sla_filter": sla_filter, "product_filter": product_filter,
            "date_from": date_from_str, "date_to": date_to_str,
            "total": 0,
            # --- START: ADD THESE DEFAULT VALUES ---
            "email_trend": 0,
            "ai_sla_met_pct": 0,
            "human_sla_met_pct": 0,
            "avg_ml_confidence": 0,
            "avg_rule_confidence": 0,
            # --- END: ADD THESE DEFAULT VALUES ---
            "volume_labels": [], "volume_data": [], "category_labels": [], "category_data": [],
            "recent_emails": [], "team_performance": [],
            "trending_products": {"up": [], "down": []}, "top_product_labels": [], "top_product_data": [],
            "top_products_chart": "", "heatmap": "", "team_performance_chart": "", "sla_trend_chart": "",
            "heatmap_data": {"series": [], "categories": []},
            "key_metrics": {
                "most_frequent": "N/A", "most_frequent_count": 0,
                "least_frequent": "N/A", "least_frequent_count": 0,
                "total_complaints": 0, "automation_rate": 0,
                "avg_ml_confidence": 0, "avg_rule_confidence": 0, "busiest_day": "N/A",
            }
        }
        return empty_context

    # --- 5. Main Data Processing (The "Happy Path") ---
    df = pd.DataFrame(logs)
    all_df = pd.DataFrame(all_logs)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    all_df["timestamp"] = pd.to_datetime(all_df["timestamp"])
    
    # ... (the rest of your processing logic is unchanged)
    
    df["SLA_Met"] = df.get("SLA_Met", False)
    if "resolution_type" not in df.columns: df["resolution_type"] = "unknown"
    ai_cases = df[df["resolution_type"] == "automated"]
    human_cases = df[df["resolution_type"] == "human"]
    ai_sla_met_pct = ai_cases["SLA_Met"].mean() * 100 if not ai_cases.empty else 100.0
    human_sla_met_pct = human_cases["SLA_Met"].mean() * 100 if not human_cases.empty else 100.0
    avg_ml_confidence = df["ml_confidence"].mean() * 100 if "ml_confidence" in df else 0
    avg_rule_confidence = df["rule_confidence"].mean() * 100 if "rule_confidence" in df else 0

    # Email Trend
    current_week = all_df[all_df["timestamp"] >= (datetime.now() - timedelta(days=7))]
    prev_week = all_df[(all_df["timestamp"] >= (datetime.now() - timedelta(days=14))) & (all_df["timestamp"] < (datetime.now() - timedelta(days=7)))]
    email_trend = ((len(current_week) - len(prev_week)) / len(prev_week) * 100) if len(prev_week) > 0 else 0

    # Key Metrics
    total_emails = len(df)
    automation_rate = (len(ai_cases) / total_emails * 100) if total_emails > 0 else 0

    # Check if category column exists (handle different possible column names)
    category_column = None
    for col in ['category', 'Category', 'predicted_category', 'email_category']:
        if col in df.columns:
            category_column = col
            break

    if not category_column:
        for col in ['category', 'Category', 'predicted_category', 'email_category']:
            if col in all_df.columns:
                category_column = col
                break
    
    if category_column:
        category_counts = df[category_column].value_counts()
        most_frequent = str(category_counts.index[0]) if len(category_counts) > 0 else "N/A"
        most_frequent_count = int(category_counts.iloc[0]) if len(category_counts) > 0 else 0
        least_frequent = str(category_counts.index[-1]) if len(category_counts) > 0 else "N/A"
        least_frequent_count = int(category_counts.iloc[-1]) if len(category_counts) > 0 else 0
        total_complaints = int(category_counts.get("complaint", 0))
    else:
        # Create empty Series if no category column exists
        category_counts = pd.Series(dtype=int)
        most_frequent = "N/A"
        most_frequent_count = 0
        least_frequent = "N/A"
        least_frequent_count = 0
        total_complaints = 0
        print(f"Warning: No category column found in df. Available columns: {df.columns.tolist()}")

    # Day of week analysis
    df['day_of_week'] = df['timestamp'].dt.day_name()
    busiest_day = df['day_of_week'].mode()[0] if len(df) > 0 else "N/A"

    key_metrics = {
        "most_frequent": most_frequent, 
        "most_frequent_count": most_frequent_count,
        "least_frequent": least_frequent, 
        "least_frequent_count": least_frequent_count,
        "total_complaints": total_complaints, 
        "automation_rate": automation_rate,
        "avg_ml_confidence": avg_ml_confidence, 
        "avg_rule_confidence": avg_rule_confidence,
        "busiest_day": busiest_day,
    }

    # --- Product Insights and Trends ---
    # This block correctly calculates the data needed for the JavaScript chart.
    insights, product_counts, _ = generate_product_insights(logs)

    # Sort top products for the bar chart
    sorted_top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    top_product_labels = [humanize_product_tag(p[0]) for p in sorted_top_products]
    top_product_data = [p[1] for p in sorted_top_products]

    # Calculate trends for the dashboard
    # Define current and previous periods
    current_period_start = datetime.now() - timedelta(days=30)
    previous_period_start = datetime.now() - timedelta(days=60)

    current_logs = [log for log in all_logs if log['timestamp'] >= current_period_start]
    previous_logs = [log for log in all_logs if previous_period_start <= log['timestamp'] < current_period_start]

    trending_products = calculate_product_trends(current_logs, previous_logs)

    # Add this code block after the lines:
    # top_product_labels = []
    # top_product_data = []

    # Process product mentions for Chart.js
    if "products_detected" in df.columns:
        # Flatten all product mentions from all emails
        all_products_mentioned = []
        for products_list in df["products_detected"].dropna():
            if isinstance(products_list, list):
                all_products_mentioned.extend(products_list)
            elif isinstance(products_list, str):
                # Handle case where products_detected is a single string
                all_products_mentioned.append(products_list)
    
        if all_products_mentioned:
            # Count product mentions
            product_counts = pd.Series(all_products_mentioned).value_counts()
        
            # Get top 5 products for Chart.js
            top_5_products = product_counts.head(5)
            top_product_labels = [humanize_product_tag(product) for product in top_5_products.index.tolist()]
            top_product_data = top_5_products.tolist()
        
            # If you don't have a humanize_product_tag function, use this simple version:
            # top_product_labels = top_5_products.index.tolist()

    # If still no data, provide some sample data to prevent errors
    if not top_product_labels:
        top_product_labels = ["No Products"]
        top_product_data = [0]


    # Team Performance
    if "resolved_by" not in df.columns: 
        df["resolved_by"] = np.nan
    team_df = df[df["resolved_by"].notna()]
    team_performance = []
    if not team_df.empty:
        performance_stats = team_df.groupby("resolved_by").agg(
            emails_handled=("resolved_by", "count"),
            avg_response_time_sec=("delay_in_sec", "mean"),
            human_sla_compliance=("SLA_Met", lambda x: x.mean() * 100)
        ).reset_index()
        performance_stats["avg_response_time"] = round(performance_stats["avg_response_time_sec"] / 3600, 1)
        performance_stats["human_sla_compliance"] = performance_stats["human_sla_compliance"].round(0)
        performance_stats["satisfaction"] = [round(random.uniform(3.8, 4.9), 1) for _ in range(len(performance_stats))]
        performance_stats["score"] = (performance_stats["human_sla_compliance"] * 0.7) + (performance_stats["satisfaction"] * 6)
        performance_stats.rename(columns={'resolved_by': 'name'}, inplace=True)
        team_performance = performance_stats.to_dict("records")

    # Volume Chart Data Calculation
    df["date"] = df["timestamp"].dt.date
    volume_counts = df.groupby("date").size()

    # --- Final Context Assembly ---
    context = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total_emails,
        "email_trend": round(email_trend, 1),
        "ai_sla_met_pct": ai_sla_met_pct,
        "human_sla_met_pct": human_sla_met_pct,
        "avg_ml_confidence": avg_ml_confidence,
        "avg_rule_confidence": avg_rule_confidence,
        "volume_labels": pd.to_datetime(volume_counts.index).strftime("%Y-%m-%d").tolist(),
        "volume_data": volume_counts.tolist(),
        "category_labels": category_counts.index.tolist() if len(category_counts) > 0 else [],
    	"category_values": category_counts.values.tolist() if len(category_counts) > 0 else [],
        "category_data": category_counts.values.tolist() if len(category_counts) > 0 else [],  # ADD THIS LINE
	"top_product_labels": top_product_labels,
        "top_product_data": top_product_data,
        "recent_emails": sorted(logs, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:5],
        "team_performance": team_performance,
        "trending_products": trending_products,
        "key_metrics": key_metrics,
        "top_products_chart": create_product_demand_chart(logs),
        "heatmap": create_category_heatmap(logs),
        "heatmap_data": prepare_heatmap_data_for_js(logs),
        "team_performance_chart": create_team_response_chart(team_performance), 
        "sla_trend_chart": create_sla_trend_chart(df),
    	"all_categories": sorted(list(all_df[category_column].unique())) if category_column and category_column in all_df.columns else [],
        "all_products": get_product_list_for_dropdown(),
        "selected_category": selected_category,
        "sla_filter": sla_filter,
        "product_filter": product_filter,
        "date_from": date_from_str,
        "date_to": date_to_str,
    }
    return context

def create_category_heatmap(logs):
    """
    Generate product-category heatmap visualization.
    This version fixes the 'unhashable type: list' error.
    """
    try:
        if not logs:
            return ""

        df = pd.DataFrame(logs)
        if df.empty or "products_detected" not in df.columns or "category" not in df.columns:
            return ""
        
        # --- THE FIX: Use .loc for robust filtering to avoid the hashing error ---
        valid_rows_mask = df['products_detected'].apply(lambda p: isinstance(p, list) and len(p) > 0)
        df = df.loc[valid_rows_mask]

        if df.empty:
            return ""
            
        df_exploded = df.explode("products_detected")
        heatmap_data = pd.crosstab(
            df_exploded["products_detected"], 
            df_exploded["category"]
        )
        
        if heatmap_data.empty:
            return ""

        top_products = df_exploded["products_detected"].value_counts().head(10).index
        heatmap_data = heatmap_data.reindex(index=top_products).dropna(how='all')
        
        if heatmap_data.empty:
            return ""
        
        heatmap_data.index = [humanize_product_tag(p) for p in heatmap_data.index]
        
        fig = plt.figure(figsize=(10, 6))
        
        sns.heatmap(
            heatmap_data, 
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            linewidths=.5
        )
        plt.title("Product-Category Heatmap")
        plt.ylabel("Products")
        plt.xlabel("Categories")
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""
    


def prepare_heatmap_data_for_js(logs):
    """
    Processes logs and returns heatmap data in a format suitable for ApexCharts.js.
    """
    if not logs:
        return {'series': [], 'categories': []}

    df = pd.DataFrame(logs)
    
    # Explode the products_detected list so each product gets its own row
    # Check if required columns exist
    if 'products_detected' not in df.columns or 'category' not in df.columns:
        print(f"Warning: Missing required columns. Available: {df.columns.tolist()}")
    return {'series': [], 'categories': []}
    
    df_exploded = df.explode('products_detected').dropna(subset=['products_detected', 'category'])
    
    if df_exploded.empty:
        return {'series': [], 'categories': []}

    # Create the crosstab to count co-occurrences
    heatmap_data = pd.crosstab(df_exploded['products_detected'], df_exploded['category'])

    # Format the data for ApexCharts
    # Series data should be: [{ name: 'Product A', data: [1, 5, 0, 3] }, { name: 'Product B', ... }]
    series_data = []
    for product, row in heatmap_data.iterrows():
        series_data.append({
            'name': humanize_product_tag(product),
            'data': row.tolist()
        })
        
    # Categories are the column headers
    category_labels = heatmap_data.columns.tolist()

    return {'series': series_data, 'categories': category_labels}
# In testapp.py

def create_team_response_chart(team_data):
    """
    Generate a modern team response time visualization matching Wayflyer style.
    """
    try:
        if not team_data:
            return ""

        # Create figure with Wayflyer-inspired styling
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_alpha(0.0)  # Transparent background
        ax.set_facecolor('#FFFFFF00')  # Transparent axes background

        # Prepare data
        df = pd.DataFrame(team_data)
        if df.empty or 'name' not in df.columns or 'avg_response_time' not in df.columns:
            return ""

        # Sort by response time (fastest first)
        df = df.sort_values('avg_response_time', ascending=True)

        # Wayflyer color scheme - blue gradient
        colors = plt.cm.Blues(np.linspace(0.4, 1, len(df)))

        # Create horizontal bars with gradient effect
        bars = ax.barh(
            df['name'],
            df['avg_response_time'],
            color=colors,
            height=0.6,  # Slightly thicker bars
            edgecolor='white',
            linewidth=0.5
        )

        # Add value labels inside bars (white text)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width - 0.2,  # Position inside the bar
                bar.get_y() + bar.get_height()/2,
                f'{width:.1f} hrs',
                ha='right',
                va='center',
                color='white',
                fontsize=11,
                fontweight='bold'
            )

        # Remove all borders and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=True)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Customize y-axis labels (employee names)
        ax.tick_params(axis='y', labelsize=12, labelcolor='#333333')

        # Add subtle grid lines
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#EEEEEE', linestyle='-', linewidth=0.5)

        # Add title with Wayflyer-style spacing
        ax.set_title('Team Response Time', 
                    pad=20, 
                    fontsize=14, 
                    fontweight='bold',
                    color='#333333')

        plt.tight_layout()

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, transparent=True, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print(f"Error generating response chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

        


def prepare_executive_report_data():
    """Prepare specialized data for executive reports with formatted insights and charts"""
    logs = list(collection.find())

    # --- START: NEW ROBUST "NO DATA" HANDLING ---
    if not logs:
        # Return a complete dictionary with default "zero" values to prevent template errors
        return {
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total': 0,
            'sla_met_pct': 0,
            'product_insights': [],
            'category_distribution': {},
            'team_data': [],
            'top_performers': [],
            'executive_charts': {
                'product_demand': "",
                'category_heatmap': "",
                'team_response': "",
                'performance_leaderboard': "",
            },
            'company_logo': None
        }
    # --- END: NEW ROBUST "NO DATA" HANDLING ---

    df = pd.DataFrame(logs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df["delay_in_sec"] = df.get("delay_in_sec", pd.Series(0, index=df.index))
    df["SLA_Met"] = df["delay_in_sec"] <= 7200

    # Overall KPIs
    total_emails = len(df)
    sla_compliance = df["SLA_Met"].mean() * 100 if not df.empty else 0
    
    # Product Insights
    insights, _, _ = generate_product_insights(logs)
    
    # Category Distribution
    category_distribution = df["category"].value_counts().to_dict()

    # Team Performance Data Calculation
    team_performance = []
    if "resolved_by" in df.columns:
        team_df = df[df["resolved_by"].notna()]
        if not team_df.empty:
            performance_stats = team_df.groupby("resolved_by").agg(
                emails_handled=("resolved_by", "count"),
                avg_response_time_sec=("delay_in_sec", "mean"),
                human_sla_compliance=("SLA_Met", lambda x: x.mean() * 100)
            ).reset_index()
            performance_stats["avg_response_time"] = round(performance_stats["avg_response_time_sec"] / 3600, 1)
            performance_stats["human_sla_compliance"] = performance_stats["human_sla_compliance"].round(0)
            performance_stats["satisfaction"] = [round(random.uniform(3.8, 4.9), 1) for _ in range(len(performance_stats))]
            performance_stats["score"] = (performance_stats["human_sla_compliance"] * 0.7) + (performance_stats["satisfaction"] * 6)
            performance_stats.rename(columns={'resolved_by': 'name'}, inplace=True)
            team_performance = performance_stats.to_dict("records")

    # Build the main context dictionary
    context = {
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total': total_emails,
        'sla_met_pct': sla_compliance,
        'product_insights': insights,
        'category_distribution': category_distribution,
        'team_data': team_performance,
        'top_performers': sorted(team_performance, key=lambda x: x.get('score', 0), reverse=True),
    }

    # Generate charts and add them to the context
    context['executive_charts'] = {
        'product_demand': create_product_demand_chart(logs),
        'category_heatmap': create_category_heatmap(logs),
        'team_response': create_team_response_chart(team_performance),
        'performance_leaderboard': create_performance_leaderboard(team_performance),
    }
    
    # Add company logo if it exists, using a path that works inside the container
    # The Docker WORKDIR is /app, so we look relative to that.
    logo_path = os.path.join(os.getcwd(), 'frontend', 'static', 'img', 'logo.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            context['company_logo'] = base64.b64encode(f.read()).decode()
    else:
        context['company_logo'] = None
        # Optional: print a warning to the logs if the logo is not found
        print(f"Warning: Logo file not found at path: {logo_path}")

    return context




def create_reply_category_chart(reply_stats):
    """Generate a pie chart for reply categories."""
    try:
        if not reply_stats:
            return ""

        labels = [stat['_id'].title() if stat['_id'] else 'Unknown' for stat in reply_stats]
        sizes = [stat['count'] for stat in reply_stats]
        
        color_map = {
            'Interested': '#28a745', 'Positive': '#17a2b8', 'Negative': '#dc3545',
            'Neutral': '#6c757d', 'Unsubscribed': '#ffc107', 'Unknown': '#adb5bd'
        }
        colors = [color_map.get(label, '#007bff') for label in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.axis('equal')
        plt.title('Reply Category Distribution', pad=20)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, transparent=True)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error generating reply category chart: {str(e)}")
        return ""


def generate_catalogue_reply():
    """Generate a reply for catalogue/general_enquiry/quotation requests"""
    return """Dear Sir / Madam,<br><br>
Thank you for showing your interest in our products. This is to confirm that we have successfully received your request.<br><br>
To provide you with more detailed information, please explore our company product catalogues here: <a href="https://drive.google.com/drive/folders/1CVo1V4iz-bwsq3Q8KNYFzmT_pXABrEYE" style="color: #1155cc;">https://drive.google.com/drive/folders/1CHIdf74z6xKLkX3CvtFhyemKVQjvlN64?usp=sharing</a><br>
We request you to review the shared details and let us know your specific requirements.<br><br>
For a smoother onboarding process, kindly fill out the attached <strong>Customer Registration Form</strong>. The information you provide will be directly recorded in our ERP system, which will help streamline future interactions and benefit both our organizations: <a href="https://forms.gle/2aWvsYC51UCCCdTj8" style="color: #1155cc;">https://forms.gle/2aWvsYC51UCCCdTj8</a><br><br>
Our sales team will connect with you shortly for further assistance.<br><br>
For immediate support, feel free to contact: 📞 <strong>Geeta Sawant:</strong> +91 9833808061 / <strong>Vishakha Parab:</strong> +91 7045100403 / <strong>Gaurita Sawant:</strong> +91 8591998713<br><br>
We look forward to serving you.<br><br>
Thanks and Regards,"""




def generate_complaint_reply():
    """Generate a reply for complaints/issues"""
    return """Dear Sir / Madam,<br><br>
We sincerely apologize for any inconvenience caused.<br><br>
We acknowledge your concern and have escalated the matter to our concerned team for detailed review. They will investigate the root cause and ensure an appropriate resolution is provided at the earliest.<br>
Our team will be in touch with you shortly to offer further assistance and address any queries you may have.<br><br>
Thank you for your patience and understanding.<br><br>
Thanks and Regards,"""

def generate_followup_reply(details, email_text):
    email_text = email_text.lower()
    dispatch_keywords = ["dispatch", "tracking", "courier", "awb", "lr copy", "shipment", "delivery", "transit"]
    has_dispatch_context = any(word in email_text for word in dispatch_keywords)
    
    # Handle reference number professionally
    ref_number = details.get("REF_NUMBER")
    if ref_number and ref_number != "N/A":
        ref_line = f"<strong>Reference: {ref_number}</strong><br><br>"
    else:
        ref_line = ""
    
    base_reply = f"""Dear Sir / Madam,<br><br>
Thank you for your follow-up regarding your recent inquiry.<br><br>
{ref_line}We wish to inform you that your request has been duly noted and escalated to the appropriate department for priority review. Our team is actively working on this matter and will provide you with a comprehensive update at the earliest.<br><br>"""
    
    dispatch_block = """In the interim, we would like to inform you that your order has been dispatched via <strong>[Courier Name]</strong>. The tracking details are as follows:<br>
<strong>Tracking ID:</strong> [Tracking ID]<br>
<strong>Expected Delivery:</strong> [Expected Timeframe]<br><br>"""
    
    closing = """We sincerely appreciate your patience and understanding. Should you require any immediate assistance, please feel free to contact us.<br><br>
Warm regards,"""
    
    full_reply = base_reply
    if has_dispatch_context:
        full_reply += dispatch_block
    full_reply += closing
    
    return full_reply



def generate_other_reply():
    return """Dear Sir / Madam,<br><br>
Thank you for reaching out to us.<br>
We have received your message and forwarded it to the concerned department for appropriate review and handling.<br>
Should any further clarification be required, our team will get in touch with you shortly.<br>
We appreciate your interest and assure you of our prompt support.<br>
Thanks and Regards,"""
