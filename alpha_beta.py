def alpha_beta(node, depth, alpha, beta, maximizingPlayer, get_children, evaluate):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    
    :param node: Current node (game state)
    :param depth: Max depth to search
    :param alpha: Best already explored option along path to maximizer (-inf initially)
    :param beta: Best already explored option along path to minimizer (+inf initially)
    :param maximizingPlayer: True if current player maximizes the score
    :param get_children: function to get possible next states
    :param evaluate: heuristic evaluation function for leaf states
    :return: (best_score, best_move)
    """
    if depth == 0 or not get_children(node):
        return evaluate(node), None
    
    if maximizingPlayer:
        maxEval = float('-inf')
        best_move = None
        for child in get_children(node):
            eval_child, _ = alpha_beta(child, depth-1, alpha, beta, False, get_children, evaluate)
            if eval_child > maxEval:
                maxEval = eval_child
                best_move = child
            alpha = max(alpha, eval_child)
            if beta <= alpha:
                break  # Beta cutoff
        return maxEval, best_move
    else:
        minEval = float('inf')
        best_move = None
        for child in get_children(node):
            eval_child, _ = alpha_beta(child, depth-1, alpha, beta, True, get_children, evaluate)
            if eval_child < minEval:
                minEval = eval_child
                best_move = child
            beta = min(beta, eval_child)
            if beta <= alpha:
                break  # Alpha cutoff
        return minEval, best_move


best_val, best_move = alpha_beta(5, 3, float('-inf'), float('inf'), True, get_children_stub, evaluate_stub)
print("Alpha-Beta Pruning result:", best_val, best_move)