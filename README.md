# Bharati AI - Sales Automation Platform

Enterprise-grade AI-powered email automation platform that streamlines sales workflows through intelligent email classification, automated response generation, and real-time analytics dashboards.

## 🚀 Features

- **AI-Powered Email Classification**: Automatically categorizes incoming emails using OpenAI API and custom ML models
- **Intelligent Response Generation**: Generates contextual draft replies to reduce manual effort and improve response time
- **Real-Time Analytics Dashboard**: Visual insights for tracking sales metrics, email performance, and team productivity
- **Campaign Management**: Create, assign, and monitor email campaigns with automated tracking
- **Lead Management**: Comprehensive lead tracking and assignment system for sales teams
- **Chrome Extension**: Browser extension for seamless Gmail integration
- **Asynchronous Processing**: Redis and Celery integration for efficient background task handling
- **Role-Based Access Control**: Separate dashboards for executives and sales personnel

## 🛠️ Tech Stack

### Backend
- **Python 3.12** - Core application language
- **Flask** - Web framework
- **MongoDB** - Primary database for storing emails, campaigns, and analytics
- **Redis** - Message broker and caching layer
- **Celery** - Distributed task queue for asynchronous processing

### AI/ML
- **OpenAI API** - GPT models for email classification and response generation
- **XGBoost** - Custom machine learning model for email categorization
- **scikit-learn** - Feature engineering and model training
- **NLTK** - Natural language processing

### Frontend
- **HTML/CSS/JavaScript** - User interface
- **Bootstrap** - Responsive design framework
- **Chart.js** - Data visualization

### Deployment
- **Ubuntu Server** - Production environment
- **Gunicorn** - WSGI HTTP server
- **Nginx** - Reverse proxy (configured separately)

## 📋 Prerequisites

- Python 3.12+
- MongoDB instance (local or MongoDB Atlas)
- Redis server
- OpenAI API key
- Google OAuth credentials (for Gmail integration)

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/r4hul-dotcom/bharati-ai.git
cd bharati-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:
```env
FLASK_APP=run_server.py
SECRET_KEY=your_secret_key_here
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MONGO_URI=your_mongodb_connection_string
FERNET_KEY=your_fernet_encryption_key
HF_TOKEN=your_huggingface_token
```

### 5. Set up Google OAuth

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable Gmail API
3. Create OAuth 2.0 credentials
4. Download credentials and save as `credentials.json` in `scripts/gmail_integration/`
5. Run authentication script:
```bash
python scripts/gmail_integration/authenticate.py
```

## 🚀 Usage

### Start Redis Server
```bash
redis-server
```

### Start Celery Worker
```bash
celery -A backend.tasks.celery_worker.celery_app worker --loglevel=info
```

### Start Celery Beat (for scheduled tasks)
```bash
celery -A backend.tasks.celery_worker.celery_app beat --loglevel=info
```

### Run the Application
```bash
python wsgi.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure
```
bharati-ai/
├── backend/
│   ├── app/              # Core application logic
│   │   ├── __init__.py
│   │   ├── auth.py       # Authentication & authorization
│   │   ├── models.py     # Database models
│   │   ├── routes.py     # API routes
│   │   └── utils.py      # Utility functions
│   ├── ml_model/         # Machine learning components
│   │   ├── classifier.py
│   │   ├── feature_engineering.py
│   │   └── saved_model/  # Trained models
│   └── tasks/            # Celery tasks
│       ├── celery_worker.py
│       └── email_tasks.py
├── frontend/
│   ├── static/           # CSS, JS, images
│   └── templates/        # HTML templates
├── chrome_extension/     # Chrome extension files
├── scripts/
│   ├── database_management/  # DB utilities
│   ├── gmail_integration/    # Gmail API integration
│   └── ml_training/          # Model training scripts
├── requirements.txt
├── .env.example
└── README.md
```

## 🔑 Key Features Explained

### Email Classification
The system uses a hybrid approach:
- OpenAI GPT models for contextual understanding
- Custom XGBoost classifier trained on company-specific data
- Confidence scoring to ensure accurate categorization

### Response Generation
- Analyzes email content and context
- Generates personalized draft responses
- Maintains brand voice consistency
- Saves sales teams hours of manual work

### Dashboard Analytics
- Real-time email processing metrics
- Campaign performance tracking
- Team productivity insights
- Lead conversion funnels

## 🔐 Security Features

- Fernet encryption for sensitive data
- JWT-based authentication
- Role-based access control (RBAC)
- Secure OAuth 2.0 integration
- Environment variable management
- Input validation and sanitization

## 🚀 Deployment

The application is production-ready and deployed on Ubuntu Server with:
- Gunicorn as WSGI server
- Nginx as reverse proxy
- Supervisor for process management
- SSL/TLS encryption

## 📊 Performance

- Processes 100+ emails per minute
- Sub-second email classification
- Automated response generation in < 3 seconds
- Handles concurrent requests with Celery workers

## 🤝 Contributing

This is a private enterprise project. For inquiries, please contact the repository owner.

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Rahul Ghole**  
Executive Engineer | AI Automation Specialist

- GitHub: [@r4hul-dotcom](https://github.com/r4hul-dotcom)
- Email: rahulghole101@gmail.com
- LinkedIn: [Add your LinkedIn URL]

## 🙏 Acknowledgments

- OpenAI for GPT API
- The Flask and Python communities
- MongoDB and Redis teams

---

**Note**: This project was developed for Bharati Fire Engineers Pvt. Ltd. to streamline their sales operations and improve team efficiency.
