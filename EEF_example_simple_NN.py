# Author: Bruce Brasseur

# I highly recommend you look at the line by line run through of this code at www.youtube.com/watch?v=bWOBxo9isNQ

import operator
import math
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import tensorflow as tf

import tensorflow.keras.backend as K

# tf.enable_eager_execution()

# This implementation works with tf 1.14.0. I was having some trouble with tf 2.0
print(tf.__version__)

# Define primitive set functions
# I casted everything to float32 a bunch just so no errors occur during long run sessions.
# I'm sure this is unesserary, but I would rather just be safe and let other people change it.

# Epsilon is added to some inputs to prevent division by zero and other errors.
def protected_div(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    result = tf.divide(a, tf.add(b, 0.00001))
    return tf.cast(result, tf.float32)


def protected_sqrt(a):
    a = tf.cast(a, tf.float32)
    result = tf.pow(abs(a), 0.5)
    return tf.cast(result, tf.float32)


def protected_pow(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    result = tf.pow(a + 0.00001, b)
    return tf.cast(result, tf.float32)


def protected_ln(a):
    a = tf.cast(a, tf.float32)
    result = tf.log(abs(a) + 0.00001)
    return tf.cast(result, tf.float32)


def protected_log(a):
    a = tf.cast(a, tf.float32)
    result = K.log(abs(a) + 0.00001)  # tf doest have log10 so we use keras backend
    return tf.cast(result, tf.float32)


def square(a):
    a = tf.cast(a, tf.float32)
    result = tf.pow(a, 2)
    return tf.cast(result, tf.float32)


def add(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    result = tf.add(a, b)
    return tf.cast(result, tf.float32)


def subtract(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    result = tf.subtract(a, b)
    return tf.cast(result, tf.float32)


def multiply(a, b):
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    result = tf.multiply(a, b)
    return tf.cast(result, tf.float32)


def exp(a):
    a = tf.cast(a, tf.float32)
    result = tf.exp(a)
    return tf.cast(result, tf.float32)


def cos(a):
    a = tf.cast(a, tf.float32)
    result = tf.cos(a)
    return tf.cast(result, tf.float32)


def sin(a):
    a = tf.cast(a, tf.float32)
    result = tf.sin(a)
    return tf.cast(result, tf.float32)


# Data preprocessing
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255.0
x_test /= 255.0

y_train_cat = tf.keras.utils.to_categorical(y_train)
y_test_cat = tf.keras.utils.to_categorical(y_test)


pset = gp.PrimitiveSet("MAIN", 2)  # Second arg is 2 because we have y and yhat
pset.addPrimitive(add, 2)
pset.addPrimitive(subtract, 2)
pset.addPrimitive(multiply, 2)
pset.addPrimitive(square, 1)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(cos, 1)
pset.addPrimitive(sin, 1)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(protected_pow, 2)
pset.addPrimitive(exp, 1)
pset.addPrimitive(protected_ln, 1)
pset.addPrimitive(protected_log, 1)

pset.renameArguments(ARG0="y_true")  # y
pset.renameArguments(ARG1="y_pred")  # yhat

pset.addEphemeralConstant("rand_int_-6_6", lambda: np.float32(random.randint(-6, 6)))
pset.addEphemeralConstant("rand_float_-6_6", lambda: np.float32(random.uniform(-6, 6)))

creator.create("FitnessMin", base.Fitness, weights=(1.0,)) # DEAP requires that weights be a tuple.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def is_invalid(individual):
    if ("y_true" in str(individual)) and ("y_pred" in str(individual)):
        return False
    return True


def eval_fitness(individual, mnist):
    # We want to maximize fitness. The best possible fitness is 100.
    # Deap requires that this function return a tuple, even if it is only 1 long.

    if is_invalid(individual):
        return (0,)

    cost_func = toolbox.compile(expr=individual)

    print("\n")
    print(individual)
    print("\n")

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(512, input_dim=784, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer="adam", loss=cost_func, metrics=["accuracy"])

    model.fit(x_train, y_train_cat, epochs=1, batch_size=128, verbose=1)

    (val_loss, val_acc) = model.evaluate(x_test, y_test_cat, verbose=0)

    del model

    K.clear_session()

    print("~~~~~~~~~~", val_acc, "~~~~~~~~~~~")

    # Just multiply val_acc by a scalar so we have nicer numbers to work with.
    fitness = 100.0 * val_acc

    return (fitness,)


toolbox.register("evaluate", eval_fitness, mnist=mnist)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)
toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
)

random.seed(10)

pop = toolbox.population(n=100)
hof = tools.HallOfFame(10)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop, log = algorithms.eaSimple(
    pop,
    toolbox,
    0.5,  # Crossover probability
    0.4,  # Mutation probability
    21,  # Generations
    stats=mstats,
    halloffame=hof,
    verbose=True,
)


print("done")

print("=====================")
print("=====================")
print("=====================")

for i in range(10):
    print(hof[i])

print("=====================")
print("=====================")
print("=====================")

