from flask import Flask, request, render_template
import json, requests
from werkzeug.utils import secure_filename
import os
from models.fourLayer import *
from boardWork import *

app = Flask(__name__, template_folder='templates/html', static_folder='static')

@app.route("/", methods=['GET', 'POST'])
def solver():
    board_path = None
    solved_filename = None

    if request.method == 'POST':
        print('posted')
        file = request.files.get('file')
        if file:
            board_path = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], board_path))


            w1, b1, w2, b2, w3, b3 = get_weights()
            board_squares = np.zeros(shape=(784, 0))
            
            _, board_squares, _ = format_board(board_path, board_squares, True)
            preds = make_predictions(board_squares, w1, b1, w2, b2, w3, b3)

            preds_string = preds2str(preds)
            print(preds_string)

            import solver
            puzzle = solver.parseInput(preds_string)
            solved, solution, numbers = solver.solve(puzzle)
            if solved:
                print(solution)
                out = output_board(numbers)
                solved_filename = 'solved_' + board_path
                cv2.imwrite(f'static/uploads/solved_{board_path}', 255*out)
            else: print('no solution')


    return render_template('home.html', filename=board_path, solved_filename=solved_filename)