from datetime import datetime, timezone
from food_quality.database import db

def get_utc_now():
    """Returns timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

class Analysis(db.Model):
    """Stores food quality prediction records."""
    __tablename__ = 'analyses'

    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    is_uncertain = db.Column(db.Boolean, default=False, nullable=False)
    explanation = db.Column(db.Text, nullable=True)
    model_version = db.Column(db.String(50), default='v1.0.0-MobileNetV2', nullable=False)
    created_at = db.Column(db.DateTime, default=get_utc_now, nullable=False, index=True)

    def to_dict(self):
        """Serialize record to dictionary for API responses."""
        return {
            'id': self.id,
            'image_filename': self.image_filename,
            'image_url': f'/uploads/{self.image_filename}',
            'predicted_label': self.predicted_label,
            'confidence': round(self.confidence, 4),
            'confidence_percentage': f"{round(self.confidence * 100, 1)}%",
            'is_uncertain': self.is_uncertain,
            'explanation': self.explanation,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat() if (self.created_at and self.created_at.tzinfo) else (self.created_at.isoformat() + 'Z' if self.created_at else ''),
            'created_at_formatted': self.created_at.strftime('%b %d, %Y - %I:%M %p')
        }

    def __repr__(self):
        return f"<Analysis id={self.id} label={self.predicted_label} conf={self.confidence:.2f}>"
