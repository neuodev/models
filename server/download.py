from flask import Blueprint, current_app, send_file
import os
from flask.json import jsonify 

bp = Blueprint('download', __name__, url_prefix='/download')

@bp.route('/<string:model>/<string:filename>')
def download(model, filename):
    file_path = os.path.join(current_app.root_path, '..', model, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': f"`{model}/{filename}` doesn't exist"})
    return send_file(os.path.join(current_app.root_path, '..', model, filename))
    
