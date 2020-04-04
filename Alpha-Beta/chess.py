import chess
import chess.svg
from IPython.display import SVG
import numpy as np

def eval_tables(piece):
    p_table = [ 0,  0,  0,  0,  0,  0,  0,  0,
                .5, .5, .5, .5, .5, .5, .5, .5,
                .1, .1, .2, .3, .3, .2, .1, .1,
                .05, .05, .1, .25, .25, .1, .05, .05,
                0,  0,  0, .2, .2,  0,  0,  0,
                .05, -.05,-.1,  0,  0,-.1, -.05, .05,
                .05, .1, .1,-.2,-.2, .1, .1,  .05,
                0,  0,  0,  0,  0,  0,  0,  0]

    n_table = [-0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5,
            -0.4,-0.2,  0,  0,  0,  0,-0.2,-0.4,
            -0.3,  0, 0.1, .15, .15, 0.1,  0,-0.3,
            -0.3, 0.05, .15, 0.2, 0.2, .15, 0.05,-0.3,
            -0.3,  0, .15, 0.2, 0.2, .15,  0,-0.3,
            -0.3, 0.05, 0.1, .15, .15, 0.1, 0.05,-0.3,
            -0.4,-0.2,  0, 0.05, 0.05,  0,-0.2,-0.4,
            -0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5]

    b_table = [-0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5,
                -0.4,-0.2,  0,  0,  0,  0,-0.2,-0.4,
                -0.3,  0, 0.1, .15, .15, 0.1,  0,-0.3,
                -0.3, 0.05, .15, 0.2, 0.2, .15, 0.05,-0.3,
                -0.3,  0, .15, 0.2, 0.2, .15,  0,-0.3,
                -0.3, 0.05, 0.1, .15, .15, 0.1, 0.05,-0.3,
                -0.4,-0.2,  0, 0.05, 0.05,  0,-0.2,-0.4,
                -0.5,-0.4,-0.3,-0.3,-0.3,-0.3,-0.4,-0.5,]

    r_table = [ 0,  0,  0,  0,  0,  0,  0,  0,
               0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05,
                -0.05,  0,  0,  0,  0,  0,  0, -0.05,
                -0.05,  0,  0,  0,  0,  0,  0, -0.05,
                -0.05,  0,  0,  0,  0,  0,  0, -0.05,
                -0.05,  0,  0,  0,  0,  0,  0, -0.05,
                -0.05,  0,  0,  0,  0,  0,  0, -0.05,
                0,  0,  0, 0.05, 0.05,  0,  0,  0]

    q_table = [-0.2,-0.1,-0.1, -0.05, -0.05,-0.1,-0.1,-0.2,
                -0.1,  0,  0,  0,  0,  0,  0,-0.1,
                -0.1,  0, 0.05, 0.05, 0.05, 0.05,  0,-0.1,
                -0.05,  0, 0.05, 0.05, 0.05, 0.05,  0, -0.05,
                0,  0, 0.05, 0.05, 0.05, 0.05,  0, -0.05,
                -0.1, 0.05, 0.05, 0.05, 0.05, 0.05,  0,-0.1,
                -0.1,  0, 0.05,  0,  0,  0,  0,-0.1,
                -0.2,-0.1,-0.1, -0.05, -0.05,-0.1,-0.1,-0.2]

    k_table = [-0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3,
                -0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3,
                -0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3,
                -0.3,-0.4,-0.4,-0.5,-0.5,-0.4,-0.4,-0.3,
                -0.2,-0.3,-0.3,-0.4,-0.4,-0.3,-0.3,-0.2,
                -0.1,-0.2,-0.2,-0.2,-0.2,-0.2,-0.2,-0.1,
                0.2, 0.2,  0,  0,  0,  0, 0.2, 0.2,
                0.2, 0.3, 0.1,  0,  0, 0.1, 0.3, 0.2]
    
    if piece == "p":
        return p_table
    if piece == "n":
        return n_table
    if piece == "b":
        return b_table
    if piece == "r":
        return r_table
    if piece == "q":
        return q_table
    if piece == "k":
        return k_table


def evaluation(board):
    # This is the simplist evaluation, just to test
    if board.is_checkmate():
        if board.turn:
            return np.inf
        else:
            return -np.inf
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    piece_total = (wp + 3*wb + 3*wn +0.05*wr + 9*wq) - (bp + 3*bb + 3*bn +0.05*br + 9*bq)

    pawn = sum([eval_tables("p")[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawn = pawn - sum([eval_tables("p")[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])

    knight = sum([eval_tables("n")[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knight = knight - sum([eval_tables("n")[chess.square_mirror(i)] for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    bishop = sum([eval_tables("b")[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishop = bishop - sum([eval_tables("b")[chess.square_mirror(i)] for i in board.pieces(chess.BISHOP, chess.BLACK)])

    rook = sum([eval_tables("r")[i] for i in board.pieces(chess.ROOK, chess.WHITE)]) 
    rook = rook - sum([eval_tables("r")[chess.square_mirror(i)] for i in board.pieces(chess.ROOK, chess.BLACK)])

    queen = sum([eval_tables("q")[i] for i in board.pieces(chess.QUEEN, chess.WHITE)]) 
    queen = queen - sum([eval_tables("q")[chess.square_mirror(i)] for i in board.pieces(chess.QUEEN, chess.BLACK)])

    king = sum([-eval_tables("k")[i] for i in board.pieces(chess.KING, chess.WHITE)]) 
    king = king - sum([eval_tables("k")[chess.square_mirror(i)] for i in board.pieces(chess.KING, chess.BLACK)])

    eval_total = pawn + knight + bishop + rook + queen + king 
    
    return piece_total + eval_total


def alpha_beta(board, alpha, beta, depth_to_go, player):
    if depth_to_go == 0:
        return evaluation(board)
    if player:
        best_score = -np.inf
        for move in board.legal_moves:
            board.push(move)
            score = alpha_beta(board, alpha, beta, depth_to_go-1, False)
            board.pop()
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if (alpha > beta):
                break
        return best_score
    else:
        best_score = np.inf
        for move in board.legal_moves:
            board.push(move)
            score = alpha_beta(board, alpha, beta, depth_to_go-1, True)
            board.pop()
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if (beta <= alpha):
                break
        return best_score



def ai_move(board, depth, color):
    bestMove = chess.Move.null()
    alpha = -np.inf
    beta = np.inf
    i = 0
    if color:
        bestVal = -np.inf
        for move in board.legal_moves:
            board.push(move)
            val = alpha_beta(board, alpha, beta, depth, color)
            if val > bestVal:
                bestVal = val
                bestMove = move
            if val > alpha:
                alpha = val
            board.pop()
            i += 1
            print("Analyzed move:", move, "to have value:", val)
        return bestMove
    else:
        bestVal = np.inf
        for move in board.legal_moves:
            board.push(move)
            val = alpha_beta(board, alpha, beta, depth, color)
            if val < bestVal:
                bestVal = val
                bestMove = move
            if val < beta:
                beta = val
            board.pop()
            i += 1
            print("Analyzed move:", move, "to have value:", val)
        return bestMove

  
board = chess.Board()
i = 0
human = int(input("HUMAN COLOR, 1 for black, 0 for white: "))
if human == 1:
    ai = True
else:
    ai = False
while not board.is_game_over():
    if i % 2 == human:
        while True:
            move = input("Your Move: ")
            try: 
                board.push_san(move)
                print(board)
                break
            except:
                print("Invalid move")
    else:
        move = ai_move(board, 4, ai)
        board.push(move)
        print(board)      
        print("COMPUTER MOVE:", move)    
    i += 1
