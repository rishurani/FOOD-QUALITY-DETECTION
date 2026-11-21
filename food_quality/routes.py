import json
import logging
from datetime import datetime, timedelta, timezone
from flask import (
    Blueprint, render_template, request, jsonify,
    send_from_directory, current_app, abort
)
from sqlalchemy import func, desc, asc

from food_quality.config import Config
from food_quality.database import db
from food_quality.models import Analysis
from food_quality.validators import (
    validate_and_process_upload, safe_delete_upload, ValidationError
)
from food_quality.predictor import FoodQualityPredictor, ModelNotLoadedError

logger = logging.getLogger(__name__)
bp = Blueprint('main', __name__)

# ============================================================================
# Web UI Pages
# ============================================================================

@bp.route('/')
def index():
    """Landing Page."""
    return render_template('index.html')

@bp.route('/dashboard')
def dashboard():
    """Dashboard with real-time statistics & visualizations."""
    return render_template('dashboard.html')

@bp.route('/analyzer')
def analyzer():
    """Interactive food quality image classification page."""
    return render_template('analyzer.html')

@bp.route('/history')
def history():
    """Paginated, searchable analysis history page."""
    return render_template('history.html')

@bp.route('/metrics')
def model_metrics():
    """Model performance, evaluation statistics & architecture."""
    return render_template('model_metrics.html')

@bp.route('/about')
def about():
    """Educational project overview, ML methodology, and limitations."""
    return render_template('about.html')

@bp.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Safely serve uploaded images."""
    return send_from_directory(Config.UPLOAD_FOLDER, filename)


# ============================================================================
# REST API Endpoints
# ============================================================================

@bp.route('/api/health', methods=['GET'])
def health_check():
    """System health check endpoint."""
    predictor = FoodQualityPredictor()
    total_count = db.session.query(func.count(Analysis.id)).scalar() or 0
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.is_model_loaded,
        'model_version': Config.MODEL_VERSION,
        'database_connected': True,
        'total_analyses': total_count,
        'uncertainty_threshold': Config.UNCERTAINTY_THRESHOLD,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@bp.route('/api/predict', methods=['POST'])
def predict():
    """
    Analyze uploaded image and predict food quality.
    Expects multipart/form-data with key 'image'.
    """
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Missing file',
            'message': 'No image file was included in the request.'
        }), 400

    file_obj = request.files['image']

    try:
        # Validate upload and securely save to disk
        safe_filename = validate_and_process_upload(file_obj)
        image_path = Config.UPLOAD_FOLDER / safe_filename

        # Predict using trained model
        predictor = FoodQualityPredictor()
        prediction_result = predictor.predict(image_path)

        # Store in SQLite database
        analysis = Analysis(
            image_filename=safe_filename,
            predicted_label=prediction_result['prediction'],
            confidence=prediction_result['confidence'],
            is_uncertain=prediction_result['is_uncertain'],
            explanation=prediction_result['explanation'],
            model_version=prediction_result['model_version']
        )
        db.session.add(analysis)
        db.session.commit()

        return jsonify({
            'success': True,
            'id': analysis.id,
            'prediction': analysis.predicted_label,
            'confidence': analysis.confidence,
            'confidence_percentage': f"{round(analysis.confidence * 100, 1)}%",
            'is_uncertain': analysis.is_uncertain,
            'explanation': analysis.explanation,
            'image_url': f'/uploads/{safe_filename}',
            'timestamp': analysis.created_at.isoformat() + 'Z',
            'model_version': analysis.model_version,
            'probabilities': prediction_result.get('probabilities', {})
        }), 201

    except ValidationError as ve:
        return jsonify({
            'success': False,
            'error': 'Validation Error',
            'message': str(ve)
        }), 400

    except ModelNotLoadedError as mne:
        # Clean up uploaded image if model fails to predict
        if 'safe_filename' in locals():
            safe_delete_upload(safe_filename)
        return jsonify({
            'success': False,
            'error': 'Model Unavailable',
            'message': str(mne),
            'model_loaded': False
        }), 503

    except Exception as e:
        logger.exception("Unexpected error in /api/predict")
        if 'safe_filename' in locals():
            safe_delete_upload(safe_filename)
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': 'Internal Processing Error',
            'message': 'An unexpected error occurred while analyzing the image.'
        }), 500

@bp.route('/api/history', methods=['GET'])
def get_history():
    """
    Retrieve paginated analysis history with search, category filtering, and sorting.
    Query parameters:
      page: int (default 1)
      per_page: int (default 12)
      search: str
      category: 'all' | 'fresh' | 'spoiled' | 'uncertain'
      sort: 'newest' | 'oldest' | 'confidence_desc' | 'confidence_asc'
    """
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 12, type=int), 100)
    search = request.args.get('search', '', type=str).strip()
    category = request.args.get('category', 'all', type=str).lower()
    sort_order = request.args.get('sort', 'newest', type=str).lower()

    query = Analysis.query

    # Search filter
    if search:
        query = query.filter(
            (Analysis.predicted_label.ilike(f'%{search}%')) |
            (Analysis.image_filename.ilike(f'%{search}%')) |
            (Analysis.explanation.ilike(f'%{search}%'))
        )

    # Category filter
    if category == 'fresh':
        query = query.filter(Analysis.predicted_label == 'Fresh', Analysis.is_uncertain == False)
    elif category == 'spoiled':
        query = query.filter(Analysis.predicted_label == 'Spoiled', Analysis.is_uncertain == False)
    elif category == 'uncertain':
        query = query.filter(Analysis.is_uncertain == True)

    # Sorting
    if sort_order == 'oldest':
        query = query.order_by(asc(Analysis.created_at))
    elif sort_order == 'confidence_desc':
        query = query.order_by(desc(Analysis.confidence))
    elif sort_order == 'confidence_asc':
        query = query.order_by(asc(Analysis.confidence))
    else:  # 'newest' is default
        query = query.order_by(desc(Analysis.created_at))

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'success': True,
        'items': [item.to_dict() for item in paginated.items],
        'pagination': {
            'page': paginated.page,
            'per_page': paginated.per_page,
            'total_items': paginated.total,
            'total_pages': paginated.pages or 1,
            'has_next': paginated.has_next,
            'has_prev': paginated.has_prev
        }
    })

@bp.route('/api/history/<int:item_id>', methods=['GET'])
def get_history_item(item_id):
    """Retrieve details of a single analysis record."""
    item = db.session.get(Analysis, item_id)
    if not item:
        return jsonify({'success': False, 'message': 'Analysis record not found'}), 404
    return jsonify({'success': True, 'item': item.to_dict()})

@bp.route('/api/history/<int:item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    """Delete a single analysis record and its corresponding uploaded image."""
    item = db.session.get(Analysis, item_id)
    if not item:
        return jsonify({'success': False, 'message': 'Analysis record not found'}), 404

    # Remove image file from storage safely
    safe_delete_upload(item.image_filename)

    # Remove from database
    db.session.delete(item)
    db.session.commit()

    return jsonify({
        'success': True,
        'message': f'Analysis record #{item_id} and associated image deleted successfully.'
    })

@bp.route('/api/history', methods=['DELETE'])
def clear_all_history():
    """Clear all analysis records and remove uploaded images."""
    try:
        items = Analysis.query.all()
        deleted_count = len(items)

        for item in items:
            safe_delete_upload(item.image_filename)
            db.session.delete(item)

        db.session.commit()
        return jsonify({
            'success': True,
            'message': 'All analysis history and associated images have been cleared.',
            'deleted_count': deleted_count
        })
    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to clear history")
        return jsonify({
            'success': False,
            'message': 'Failed to clear history records.'
        }), 500

@bp.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """
    Calculate and return live dashboard metrics from the SQLite database.
    """
    total = db.session.query(func.count(Analysis.id)).scalar() or 0
    fresh = db.session.query(func.count(Analysis.id)).filter(
        Analysis.predicted_label == 'Fresh', Analysis.is_uncertain == False
    ).scalar() or 0
    spoiled = db.session.query(func.count(Analysis.id)).filter(
        Analysis.predicted_label == 'Spoiled', Analysis.is_uncertain == False
    ).scalar() or 0
    uncertain = db.session.query(func.count(Analysis.id)).filter(
        Analysis.is_uncertain == True
    ).scalar() or 0

    # Calculate average confidence
    avg_conf = db.session.query(func.avg(Analysis.confidence)).scalar() or 0.0

    # Recent 5 analyses
    recent_query = Analysis.query.order_by(desc(Analysis.created_at)).limit(5).all()
    recent = [item.to_dict() for item in recent_query]

    # Daily activity for the past 7 days
    seven_days_ago = datetime.now(timezone.utc).date() - timedelta(days=6)
    daily_stats = []
    
    for i in range(7):
        target_date = seven_days_ago + timedelta(days=i)
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date, datetime.max.time())

        day_total = db.session.query(func.count(Analysis.id)).filter(
            Analysis.created_at >= start_dt, Analysis.created_at <= end_dt
        ).scalar() or 0

        day_fresh = db.session.query(func.count(Analysis.id)).filter(
            Analysis.created_at >= start_dt, Analysis.created_at <= end_dt,
            Analysis.predicted_label == 'Fresh'
        ).scalar() or 0

        day_spoiled = db.session.query(func.count(Analysis.id)).filter(
            Analysis.created_at >= start_dt, Analysis.created_at <= end_dt,
            Analysis.predicted_label == 'Spoiled'
        ).scalar() or 0

        daily_stats.append({
            'date': target_date.strftime('%b %d'),
            'total': day_total,
            'fresh': day_fresh,
            'spoiled': day_spoiled
        })

    return jsonify({
        'success': True,
        'total_analyses': total,
        'fresh_count': fresh,
        'spoiled_count': spoiled,
        'uncertain_count': uncertain,
        'fresh_percentage': round((fresh / total * 100), 1) if total > 0 else 0,
        'spoiled_percentage': round((spoiled / total * 100), 1) if total > 0 else 0,
        'average_confidence': round(avg_conf, 4),
        'average_confidence_percentage': f"{round(avg_conf * 100, 1)}%",
        'recent_analyses': recent,
        'daily_activity': daily_stats
    })

@bp.route('/api/model/metrics', methods=['GET'])
def get_model_metrics():
    """
    Return machine learning model metrics calculated from held-out test evaluation.
    """
    if not Config.METRICS_PATH.exists():
        return jsonify({
            'evaluated': False,
            'message': (
                "Model has not been evaluated yet. Run 'python train_model.py' or "
                "'python evaluate_model.py' to generate held-out test evaluation metrics."
            )
        })

    try:
        with open(Config.METRICS_PATH, 'r') as f:
            data = json.load(f)
        return jsonify({
            'evaluated': True,
            'metrics': data
        })
    except Exception as e:
        logger.error(f"Error reading metrics file: {e}")
        return jsonify({
            'evaluated': False,
            'message': f"Error reading evaluation metrics: {str(e)}"
        }), 500
