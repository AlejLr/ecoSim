def lotka_volterra(t, y, alpha, beta, delta, gamma):
    prey, predator = y
    dprey_dt = alpha * prey - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]


def logistic_predator_prey(t, y, r, K, beta, delta, gamma):
    prey, predator = y
    dprey_dt = r * prey * (1 - prey / K) - beta * prey * predator
    dpredator_dt = delta * prey * predator - gamma * predator
    return [dprey_dt, dpredator_dt]