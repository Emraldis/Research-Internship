import utils

import sys

from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklvq import GMLVQ

path = 'C:/Users/alfie/Documents/Schoolwork/Research Internship/Cushing - data for Michael.xlsx'

includeControl = False
showSteps = False
printStepReports = False

DEBUG = {
    "parsingData":False,

}

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

parsed = utils.fileParser(path)
test_sets = []
training_runs = 1
lambdaMatrixAvg = []


temp_data = []
for entry in parsed.data:
    if not includeControl:
        if entry.label != 0:
            temp_data.append(entry)
    else:
        temp_data.append(entry)
data_set = utils.dataSet(temp_data, training_runs)

if DEBUG["parsingData"]:
    for i in range(len(data_set.data)):
        print("Label: " + str(data_set.labels[i]) + " | Data: " + str(data_set.data[i]))

feature_names = parsed.dataNames


scaler = StandardScaler()

full_data = scaler.fit_transform(data_set.data)
full_labels = data_set.labels

for i in range(training_runs):

    logger = utils.processLogger()

    [train_data, train_labels], [test_data, test_labels] = data_set.get_split_data_set(i)

    label_names = data_set.labelNames

    data = scaler.fit_transform(train_data)
    labels = train_labels

    model = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="identity",
        #activation_type="swish",
        #activation_params={"beta": 2},
        solver_type="waypoint-gradient-descent",
        solver_params={
            "max_runs": 200,
            "k": 2,
            "step_size": np.array([0.1, 0.05]),
            "callback": logger},
        random_state=1428,
    )

    pipeline = make_pipeline(scaler, model)

    randForest = RandomForestClassifier(max_depth=200, random_state=0)

    #pipeline.fit(data, labels)
    model.fit(data, labels)
    randForest.fit(data, labels)


    data = scaler.fit_transform(test_data)
    labels = test_labels

    #predicted_labels = pipeline.predict(data)
    predicted_labels = model.predict(data)

    relevance_matrix = model.lambda_
    lambdaMatrixAvg.append(relevance_matrix)

    if printStepReports:
        print("GMLVQ:\n" + classification_report(labels, predicted_labels))

    predicted_tree = randForest.predict(data)
    if printStepReports:
        print("Random Forest:\n" + classification_report(labels, predicted_tree))

    for state in logger.states:
        print("STATE")
        print(state)

    if showSteps:

        iteration, fun = zip(*[(state["nit"], state["fun"]) for state in logger.states])
        tfun, nfun = zip(*[(state["tfun"], state["nfun"]) for state in logger.states])

        ax = plt.axes()

        ax.set_title("Learning Curves (Less is better)")
        ax.plot(iteration, nfun)
        ax.plot(iteration, tfun)
        _ = ax.legend(["Cost of regular gradient update", "Cost of average gradient update"])

        plt.show()
        if not includeControl:
            label_score = model.decision_function(data)

            fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=label_score, pos_label=2, drop_intermediate=True)
            roc_auc = roc_auc_score(y_true=labels, y_score=label_score)

            #  Sometimes it is good to know where the Nearest prototype classifier is on this curve. This can
            #  be computed using the confusion matrix function from sklearn.
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=model.predict(data)).ravel()

            # The tpr and fpr of the npc are then given by:
            npc_tpr = tp / (tp + fn)
            npc_fpr = fp / (fp + tn)

            fig, ax = plt.subplots()
            fig.suptitle("Receiver operating characteristic ")
            # Plot the ROC curve
            ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC AUC = {:.3f}".format(roc_auc))
            # Plot the random line
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            # Plot the NPC classifier
            ax.plot(npc_fpr, npc_tpr, color="green", marker="o", markersize="12")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(False)

            plt.show()

avg_relevance_matrix = lambdaMatrixAvg[0]

if len(lambdaMatrixAvg) > 1:
    for i in range(1,len(lambdaMatrixAvg),1):
        avg_relevance_matrix = avg_relevance_matrix + lambdaMatrixAvg[i]

avg_relevance_matrix = avg_relevance_matrix/len(lambdaMatrixAvg)

fig, ax = plt.subplots()
fig.suptitle("Relevance Matrix Diagonal")
ax.bar(feature_names, np.diagonal(avg_relevance_matrix))
ax.set_ylabel("Weight")
ax.grid(False)

plt.show()

fig, ax = plt.subplots()
fig.suptitle("Eigenvalues")
ax.bar(range(0, len(model.eigenvalues_)), model.eigenvalues_)
ax.set_ylabel("Weight")
ax.grid(False)

plt.show()

fig, ax = plt.subplots()
fig.suptitle("Random Forest Feature Importances")
ax.bar(feature_names, randForest.feature_importances_)
ax.set_ylabel("Feature Importance")
ax.grid(False)

plt.show()

data = full_data
labels = full_labels

transformed_data = model.transform(data, scale=True)

x_d = transformed_data[:, 0]
y_d = transformed_data[:, 1]

transformed_model = model.transform(model.prototypes_, scale=True)

x_m = transformed_model[:, 0]
y_m = transformed_model[:, 1]

# Plot
fig, ax = plt.subplots()
fig.suptitle("Discriminative projection Steroid Metabolomics Data and GMLVQ prototypes")
if includeControl:
    colors = ["blue", "red", "green"]
else:
    colors = ["blue", "red"]
for i, cls in enumerate(model.classes_):
    ii = cls == labels
    ax.scatter(
        x_d[ii],
        y_d[ii],
        c=colors[i],
        s=100,
        alpha=0.5,
        edgecolors="white",
        label = label_names[i]
    )
ax.scatter(x_m, y_m, c=colors, s=180, alpha=0.8, edgecolors="black", linewidth=2.0)
ax.set_xlabel("First eigenvector")
ax.set_ylabel("Second eigenvector")
ax.legend()
ax.grid(True)

plt.show()
