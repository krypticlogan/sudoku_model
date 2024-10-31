from flask import Flask, request, render_template
import json, requests
from werkzeug.utils import secure_filename
import os
from models.fourLayer import *
from boardWork import *

app = Flask(__name__, template_folder='templates/html', static_folder='static')
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# # Check if the uploaded file is an allowed type
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

            # solved_filename = 'solved_' + board_path

            w1, b1, w2, b2, w3, b3 = get_weights()
            board_squares = np.zeros(shape=(784, 0))
            
            # board_path = 'boards/board.png'
            grids, board_squares, _ = format_board(board_path, board_squares, True)
            # print(board_squares.shape)
            # print(list(grids))
            preds = make_predictions(board_squares, w1, b1, w2, b2, w3, b3)
            # print(preds)
            # grids = setup(preds)
            # print(grids)
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
            # preds_text = board_to_text






        # if request.form['submit-button'] == 'solve':
            # print('solve')

        # else: print('dont')
    return render_template('home.html', filename=board_path, solved_filename=solved_filename)