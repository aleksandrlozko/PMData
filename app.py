
from flask import Flask, request, jsonify, make_response

import pandas as pd
import json


app = Flask(__name__)


@app.route('/main', methods=['GET'])
def main():
    received_keys = sorted(list(request.form.keys()))
    if len(received_keys) > 1 or 'data' not in received_keys:
        err = 'Wrong request keys'
        return make_response(jsonify(error=err), 400)



if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8000)