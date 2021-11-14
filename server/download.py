from flask import Blueprint, current_app, send_file
import os
from flask.json import jsonify 

bp = Blueprint('download', __name__, url_prefix='/download')

@bp.route('/<string:model>/<string:filename>')
def download(model, filename):
    return send_file(os.path.join(current_app.root_path, '..', model, filename))
    
