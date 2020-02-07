from flask import Flask
from flask import render_template, request, jsonify
import re
app = Flask(__name__)
from web_falsk import db_connection as db_conn

'''
@app.route('/ai_calc', methods=['post'])
def ai_calc():
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']
    c = CalculatorController(num1, num2, opcode)
    result = c.calc()
    render_params ={}
    render_params['result'] = int(result)
    print('app.py에 출력 된 덧셈결과 ; {}'.format(result))
    return render_template('ai_calc.html', **render_params)
'''

@app.route('/')
def index():
    conn = db_conn.conn()
    curs = conn.cursor()
    sql = "SELECT * FROM MOVIE_LIST"
    curs.execute(sql)
    rows = curs.fetchall()
    sql_lists = list()
    for i,e,a in rows:
        sql_lists.append({
                           'movie_id':i
                         , "title":e
                         , "img_title":a
                        })

    print(sql_lists)
    conn.close()
    return render_template('index.html', sql_list=sql_lists)

@app.route('/detail/<movie_id>')
def detail(movie_id):
    print('detail')
    print(movie_id)
    conn = db_conn.conn()
    curs = conn.cursor()
    sql = "SELECT * FROM MOVIE_LIST WHERE MOVIE_ID = " + movie_id
    curs.execute(sql)
    rows = curs.fetchall()
    sql_lists = list()
    for i,e,a in rows:
          dict_result = {
                           'movie_id':i
                         , "title":e
                         , "img_title":a
                        }

    print(dict_result)
    conn.close()
    return render_template('detail.html', dict_results=dict_result, title=dict_result['title'])


'''
@app.route('/login', methods=['post'])
def login():
    userid = request.form['userid']
    password = request.form['password']
    ctrl = MemberController()
#    ctrl.create_table()
    view = ctrl.login(userid, password)
    return render_template(view)


#주소부분을 변수처리
@app.route('/move/<path>')
def move(path):
    return render_template('{}.html'.format(path))

@app.route('/cabbage', methods=['post'])
def cabbage():
    avg_temp = request.form['avg_temp']
    min_temp = request.form['min_temp']
    max_temp = request.form['max_temp']
    rain_fall = request.form['rain_fall']
    ctrl = CabbageController(avg_temp, min_temp, max_temp, rain_fall)
    result = ctrl.service()
    render_params = {}
    render_params['result'] = result
    return render_template('cabbage.html', **render_params)
'''


if __name__ == "__main__":
    app.run()