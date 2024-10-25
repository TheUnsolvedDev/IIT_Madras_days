from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    elif request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if password == 'cs3205':
            return redirect(url_for('index', username=username))
        else:
            flash('Invalid credentials. Please try again.')
            return redirect(url_for('login_page'))

@app.route('/index')
def index():
    username = request.args.get('username', 'Guest')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html', username=username, current_time=current_time)


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
