from flask_login import UserMixin
from datetime import datetime
from bson import ObjectId

class User(UserMixin, dict):
    def __init__(self, user_dict):
        super().__init__(user_dict)
    
    @property
    def id(self):
        return str(self.get('_id'))
    
    def get_id(self):
        return str(self.get('_id'))
    
    @property
    def email(self):
        return self.get('email')
    
    @property
    def is_active(self):
        return self.get('is_active', True)
    
    @property
    def role(self):
        return self.get('role', 'user')
    
    @property
    def name(self):
        return self.get('name', self.get('email', 'Unknown'))

class CampaignReply:
    """MongoDB-based model for campaign replies"""
    
    def __init__(self, reply_dict=None):
        self.data = reply_dict or {}
    
    @property
    def id(self):
        return str(self.data.get('_id', ''))
    
    @property
    def message_id(self):
        return self.data.get('message_id')
    
    @property
    def thread_id(self):
        return self.data.get('thread_id')
    
    @property
    def campaign_id(self):
        return self.data.get('campaign_id')
    
    @property
    def sender_email(self):
        return self.data.get('sender_email')
    
    @property
    def subject(self):
        return self.data.get('subject')
    
    @property
    def body(self):
        return self.data.get('body')
    
    @property
    def received_at(self):
        return self.data.get('received_at')
    
    @property
    def status(self):
        return self.data.get('status', 'unread')
    
    def to_dict(self):
        """Convert to dictionary for MongoDB insertion"""
        return {
            'message_id': self.message_id,
            'thread_id': self.thread_id,
            'campaign_id': self.campaign_id,
            'sender_email': self.sender_email,
            'subject': self.subject,
            'body': self.body,
            'received_at': self.received_at or datetime.utcnow(),
            'status': self.status
        }
    
    @staticmethod
    def create(replies_collection, **kwargs):
        """Create a new reply in MongoDB"""
        reply_data = {
            'message_id': kwargs.get('message_id'),
            'thread_id': kwargs.get('thread_id'),
            'campaign_id': kwargs.get('campaign_id'),
            'sender_email': kwargs.get('sender_email'),
            'subject': kwargs.get('subject'),
            'body': kwargs.get('body'),
            'received_at': kwargs.get('received_at', datetime.utcnow()),
            'status': kwargs.get('status', 'unread')
        }
        result = replies_collection.insert_one(reply_data)
        reply_data['_id'] = result.inserted_id
        return CampaignReply(reply_data)
    
    @staticmethod
    def find_by_message_id(replies_collection, message_id):
        """Find a reply by message_id"""
        reply_data = replies_collection.find_one({'message_id': message_id})
        return CampaignReply(reply_data) if reply_data else None
    
    @staticmethod
    def find_all(replies_collection, **filters):
        """Find all replies matching filters"""
        replies = replies_collection.find(filters)
        return [CampaignReply(reply) for reply in replies]
