# -*- encoding: utf-8 -*-

from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound

import pandas as pd


@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        if template == 'baseline.html':
            return baseline()

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound as log_erro404:
        print(log_erro404)
        return render_template('home/page-404.html'), 404

    except Exception as log_erro500:
        print(log_erro500)
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


@blueprint.route('/prever', methods=['POST'])
@login_required
def prever():
    data = request.form['data']
    texto = request.form['texto']
    number = request.form['number']

    # Faça algo com os dados capturados, como realizar uma previsão

    # Retorne a resposta da rota em formato HTML
    return data


# Baseline
@blueprint.route('/baseline')
@login_required
def baseline():
    from ia.preprocessing import df_resumo, hd_db

    return render_template('home/baseline.html', df_resumo=df_resumo, hd_db=hd_db)

