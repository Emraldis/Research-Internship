import utils

import sys

import sklvq

from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklvq import GMLVQ
from sklearn.model_selection import cross_val_score, RepeatedKFold

includeControl = True
healthyUnhealthy = True
showSteps = False
printStepReports = False
showPlots = False
savePlots = True

plot_label_size = 24

font = {'size'   : plot_label_size}

matplotlib.rc('font', **font)

DEBUG = {
    "parsingData":False,
    "logger_states":False,
    "avg_ROC_AUC":False,
    "Include Individual AUC Entries": True,
    "Include Individual Cost Entries": True,
}

path = 'C:/Users/alfie/Documents/Schoolwork/Research Internship/'
inputFilePath = path + 'Cushing - data for Michael.xlsx'

if includeControl:
    outputFilePath = path + "Saved_figures_control/"
    outputFileNameAddOn = "_Control_Included"
if not includeControl:
    outputFilePath = path + "Saved_figures_no_control/"
    outputFileNameAddOn = "_No_Control"
if healthyUnhealthy:
    outputFilePath = path + "Saved_figures_healty_unhealthy/"
    outputFileNameAddOn = "_Healthy_Unhealthy"

matplotlib.rc("xtick", labelsize="small")
matplotlib.rc("ytick", labelsize="small")

parsed = utils.fileParser(inputFilePath)
test_sets = []
training_runs = 20
lambdaMatrixAvg = []
aucAvg = []
y_score_sum = []
labels_sum = []
label_score_sum = []
tfunAvg = []
nfunAvg = []
avg_prototypes = []

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

if (includeControl and healthyUnhealthy) or not includeControl:
    collect_auc_data = True
else:
    collect_auc_data = False

for i in range(training_runs):

    logger = utils.processLogger(data_set.data, data_set.labels, collect_auc_data)

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
            "max_runs": 100,
            "k": 2,
            "step_size": np.array([0.1, 0.05]),
            "callback": logger},
        random_state=1428,
    )

    pipeline = make_pipeline(scaler, model)

    #print(i)
    #repeated_10_fold = RepeatedKFold(n_splits=10, n_repeats=10)
    #accuracy = cross_val_score(pipeline, data, labels, cv=repeated_10_fold, scoring="accuracy")

    randForest = RandomForestClassifier(max_depth=100, random_state=0)

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

    if DEBUG["logger_states"]:
        for state in logger.states:
            print("STATE MODEL")
            print(state["model"])
            print("STATE OMEGA MATRIX DIAGONAL")
            print(np.diagonal(state["omega_matrix"]))
            print("STATE LAMBDA MATRIX DIAGONAL")
            print(np.diagonal(GMLVQ._compute_lambda(state["omega_matrix"])))
            print("STATE PROTOTYPES")
            print(state["prototypes"])
        print("RELEVANCE MATRIX")
        print(np.diagonal(relevance_matrix))

    if DEBUG["avg_ROC_AUC"]:
        print("ROC AUC")
        for auc in logger.auc:
            print(auc)

        fig.set_size_inches(19,10)
        fig, ax = plt.subplots(constrained_layout=True)
        #ax.set_title("AUC over time")
        ax.plot(logger.auc)

        if showPlots:
            plt.show()
        if savePlots:
            fileName = outputFilePath + "ROC_AUC_" + str(i) + outputFileNameAddOn + ".png"
            plt.savefig(fileName)

    if (includeControl and healthyUnhealthy) or not includeControl:
        aucAvg.append(logger.auc)

    avg_prototypes.append(model.prototypes_)

    tfun, nfun = zip(*[(state["tfun"], state["nfun"]) for state in logger.states])
    tfunAvg.append(tfun)
    nfunAvg.append(nfun)

    label_score = model.decision_function(data)
    y_score = model.predict(data)

    label_score_sum = np.append(label_score_sum, label_score)
    y_score_sum = np.append(y_score_sum, y_score)
    labels_sum = np.append(labels_sum, labels)

    if showSteps:

        iteration, fun = zip(*[(state["nit"], state["fun"]) for state in logger.states])

        fig, ax = plt.subplots(constrained_layout=True)
        fig.set_size_inches(19,10)
        #ax.set_title("Learning Curves (Less is better)")
        ax.plot(iteration, nfun)
        ax.plot(iteration, tfun)
        _ = ax.legend(["Cost of regular gradient update", "Cost of average gradient update"])

        if showPlots:
            plt.show()
        if savePlots:
            fileName = outputFilePath + "Cost_Function_" + str(i) +  outputFileNameAddOn + ".png"
            plt.savefig(fileName)
        if not includeControl:
            fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=label_score, pos_label=2, drop_intermediate=True)
            roc_auc = roc_auc_score(y_true=labels, y_score=label_score)

            #  Sometimes it is good to know where the Nearest prototype classifier is on this curve. This can
            #  be computed using the confusion matrix function from sklearn.
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=model.predict(data)).ravel()

            # The tpr and fpr of the npc are then given by:
            npc_tpr = tp / (tp + fn)
            npc_fpr = fp / (fp + tn)

            fig, ax = plt.subplots(constrained_layout=True)
            #fig.suptitle("Receiver operating characteristic ")
            fig.set_size_inches(19,10)
            # Plot the ROC curve
            ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC AUC = {:.3f}".format(roc_auc))
            # Plot the random line
            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            # Plot the NPC classifier
            ax.plot(npc_fpr, npc_tpr, color="green", marker="o", markersize="12")
            ax.tick_params(axis='both', labelsize=plot_label_size)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(False)

            if showPlots:
                plt.show()
            if savePlots:
                fileName = outputFilePath + "ROC_Curve" + str(i) +  outputFileNameAddOn + ".png"
                plt.savefig(fileName)

if not includeControl:
    fpr, tpr, thresholds = roc_curve(y_true=labels_sum, y_score=label_score_sum, pos_label=2, drop_intermediate=True)
    roc_auc = roc_auc_score(y_true=labels_sum, y_score=label_score_sum)
    tn, fp, fn, tp = confusion_matrix(y_true=labels_sum, y_pred=y_score_sum).ravel()
    npc_tpr = tp / (tp + fn)
    npc_fpr = fp / (fp + tn)
    if not showPlots:
        fig, ax = plt.subplots(constrained_layout=True)
        #fig.suptitle("Receiver Operating Characteristics")
        fig.set_size_inches(19,10)
        # Plot the ROC curve
        ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC AUC = {:.3f}".format(roc_auc))
        # Plot the random line
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        # Plot the NPC classifier
        ax.plot(npc_fpr, npc_tpr, color="green", marker="o", markersize="12")
        ax.tick_params(axis='both', labelsize=plot_label_size)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(False)

    if showPlots:
        plt.show()
    if savePlots:
        fileName = outputFilePath + "Average_ROC_Curve" + outputFileNameAddOn + ".png"
        plt.savefig(fileName)

disp = ConfusionMatrixDisplay(confusion_matrix(y_true=labels_sum, y_pred=y_score_sum, normalize='true'))
fig, ax = plt.subplots(constrained_layout=True)
disp.plot(ax=ax, cmap='Blues')

label_font = {'size':plot_label_size}
ax.set_xlabel('Predicted labels', fontdict=label_font);
ax.set_ylabel('Observed labels', fontdict=label_font);

title_font = {'size':plot_label_size}
ax.set_title('Confusion Matrix', fontdict=title_font);

ax.tick_params(axis='both', which='major', labelsize=plot_label_size)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Confusion_matrix" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

if (includeControl and healthyUnhealthy) or not includeControl:
    avg_auc = utils.calc_array_average(aucAvg)

    fig, ax = plt.subplots(constrained_layout=True)
    #ax.set_title("AUC Over Time")
    fig.set_size_inches(19,10)
    ax.tick_params(axis='both', labelsize=plot_label_size)
    if DEBUG["Include Individual AUC Entries"]:
        for entry in aucAvg:
            ax.plot(entry, linestyle='dotted', alpha = 0.2)

    ax.plot(avg_auc)
    if showPlots:
        plt.show()
    if savePlots:
        fileName = outputFilePath + "ROC_AUC_Avg_over_time" + outputFileNameAddOn + ".png"
        plt.savefig(fileName)

avg_nfun = utils.calc_array_average(nfunAvg)

nfun = avg_nfun
fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(19,10)
ax.tick_params(axis='both', labelsize=plot_label_size)
#ax.set_title("Average Cost Function")
ax.plot(nfun)
if DEBUG["Include Individual Cost Entries"]:
    for entry in nfunAvg:
        ax.plot(entry, linestyle='dotted', alpha = 0.2)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Cost_Function_Avg" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

avg_relevance_matrix = utils.calc_array_average(lambdaMatrixAvg)
std_dev_relevance = utils.calc_array_std_dev(np.diagonal(lambdaMatrixAvg))
avg_lambda_eigenvalues, v = np.linalg.eig(avg_relevance_matrix)

fig, ax = plt.subplots(constrained_layout=True)
fig.set_size_inches(7,4)
ax.tick_params(axis='both', labelsize=plot_label_size)
ax.bar(range(0, len(avg_lambda_eigenvalues)), avg_lambda_eigenvalues)
ax.grid(False)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Average_lambda_matrix_eigenvalues" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

fig, ax = plt.subplots(constrained_layout=True)
#fig.suptitle("Average Relevance Matrix Diagonal")
fig.set_size_inches(19,10)
ax.tick_params(axis='y', labelsize=plot_label_size)
ax.tick_params(axis='x', rotation=45, labelsize=plot_label_size)
ax.bar(feature_names, np.diagonal(avg_relevance_matrix), yerr = std_dev_relevance)
ax.set_ylabel("Diagonal Relevances")
ax.grid(False)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Average_Relevance" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

avg_prototypes = utils.calc_array_average(avg_prototypes)

fig, ax = plt.subplots(len(avg_prototypes), constrained_layout=True)

#

fig.set_size_inches(19,10)
i = 0
for entry in avg_prototypes:
    print(label_names[i])
    ax[i].tick_params(axis='y', labelsize=plot_label_size)
    ax[i].tick_params(axis='x', rotation=45, labelsize=plot_label_size)
    if (i != len(avg_prototypes) - 1):
        ax[i].set_xticklabels([])
    ax[i].bar(feature_names, entry)
    ax[i].set_title(label_names[i])
    ax[i].grid(True, axis='y')
    i += 1

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Average_Prototypes" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

fig, ax = plt.subplots(constrained_layout=True)
#fig.suptitle("Eigenvalues")
fig.set_size_inches(19,10)
ax.tick_params(axis='both', labelsize=plot_label_size)
ax.bar(range(0, len(model.eigenvalues_)), model.eigenvalues_)
ax.set_ylabel("Weight")
ax.grid(False)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Sample_eigenvalues" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)

fig, ax = plt.subplots(constrained_layout=True)
#fig.suptitle("Random Forest Feature Importances")
fig.set_size_inches(19,10)
ax.tick_params(axis='y', labelsize=plot_label_size)
ax.tick_params(axis='x', rotation=45, labelsize=plot_label_size)
ax.bar(feature_names, randForest.feature_importances_)
ax.set_ylabel("Feature Importance")
ax.grid(False)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Average_forest_feature_importances" + outputFileNameAddOn + ".png"
    plt.savefig(fileName)

data = full_data
labels = full_labels

transformed_data = model.transform(data, scale=True)

x_d = transformed_data[:, 0]
y_d = transformed_data[:, 1]

transformed_model = model.transform(model.prototypes_, scale=True)

x_m = transformed_model[:, 0]
y_m = transformed_model[:, 1]

# Plot
fig, ax = plt.subplots(constrained_layout=True)
#fig.suptitle("Discriminative projection Steroid Metabolomics Data and GMLVQ prototypes")
fig.set_size_inches(19,10)
if len(model.classes_) == 3:
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
ax.tick_params(axis='both', labelsize=plot_label_size)
ax.scatter(x_m, y_m, c=colors, s=180, alpha=0.8, edgecolors="black", linewidth=2.0)
#ax.set_xlabel("First eigenvector")
#ax.set_ylabel("Second eigenvector")
ax.legend(prop={'size': (plot_label_size * 1.5)})
ax.grid(True)

if showPlots:
    plt.show()
if savePlots:
    fileName = outputFilePath + "Transformed_data" +  outputFileNameAddOn + ".png"
    plt.savefig(fileName)
