import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




### GENERAL PLOTS OF THE DATA ###




url = "https://raw.githubusercontent.com/oskarwiese/fagprojekt/master/compas/compas-scores-raw.csv?token=ANI5TF63X6C2QT7TZE2WLL26JB2YS"
data = pd.read_csv(url)
#print(data.head)
#print(data.columns)


# Check if there are any missing values
#print(np.count_nonzero(data["IsDeleted"] == 1))

def plots():
    # Show distribution of different ethnicities and sexes
    sb.countplot(x = "Ethnic_Code_Text", data = data)
    plt.show()
    sb.countplot(x = "Sex_Code_Text", data = data)
    plt.show()
    sb.countplot(x = "Language", data = data)
    plt.show()
    
    # Showing the distribution of the raw and decile values
    plt.xlabel("Raw value")
    plt.ylabel("Frequency")
    plt.title("Visualization of the values")
    plt.hist(data["RawScore"])
    plt.show()
    
    plt.xlabel("Decile value")
    plt.ylabel("Frequency")
    plt.title("Visualization of the decile values")
    plt.hist(data["DecileScore"])
    plt.show()
    
    
    #sb.countplot(x = "RawScore", hue = "Ethnic_Code_Text", data = data)
    #plt.show()
    
    # Indication that some black people might get higher sentences that white people
    sb.countplot(x = "DecileScore", hue = "Ethnic_Code_Text", data = data)
    plt.show()
    
    sb.countplot(x = "ScoreText", hue = "Ethnic_Code_Text", data = data)
    plt.show()
#plots(True)




### DATA PRE-PROCESSING AND PREPARATION ###


#categoricals = ["Agency_Text", "Sex_Code_Text", "Ethnic_Code_Text", "ScaleSet_ID", "AssessmentReason", "Language", "LegalStatus", "CustodyStatus", "MaritalStatus", "RecSupervisionLevel", "DateOfBirth", "Screening_Date"]
categoricals = ["Agency_Text", "Sex_Code_Text", "Ethnic_Code_Text", "ScaleSet_ID", "AssessmentReason", "Language", "LegalStatus", "CustodyStatus", "MaritalStatus", "RecSupervisionLevel"]

# Changing date of birth into age, as this should work better in a neural network
ages = [None] * len(data["DateOfBirth"])
for i in range(len(data["DateOfBirth"])):
    ages[i] = 20 +(100 - int(data["DateOfBirth"][i].split("/")[2]))
data["DateOfBirth"] = ages

numericals = ["DateOfBirth"]


#outputs = ["DecileScore", "ScoreText"]


outputs = ["DecileScore"]
data["DecileScore"] = data["DecileScore"].replace(-1, 0)
data["DecileScore"] = data["DecileScore"].astype("category")
print(data["DecileScore"].cat.categories)


for category in categoricals:
    data[category] = data[category].astype("category")


# Preparing data for pytorch
# Lortemåde at lave categoricals om til tensors
a = data[categoricals[0]].cat.codes.values
b = data[categoricals[1]].cat.codes.values
c = data[categoricals[2]].cat.codes.values
d = data[categoricals[3]].cat.codes.values
e = data[categoricals[4]].cat.codes.values
f = data[categoricals[5]].cat.codes.values
g = data[categoricals[6]].cat.codes.values
h = data[categoricals[7]].cat.codes.values
i = data[categoricals[8]].cat.codes.values
j = data[categoricals[9]].cat.codes.values
#k = X[categoricals[10]].cat.codes.values
#l = X[categoricals[11]].cat.codes.values

#X = np.stack([a,b,c,d,e,f,g,h,i,j,k,l], 1)
Xcat = np.stack([a,b,c,d,e,f,g,h,i,j], 1)
Xcat = torch.tensor(Xcat, dtype=torch.int64)


# Converting the numerical values to a tensor
Xnum = np.stack([data[col].values for col in numericals], 1)
Xnum = torch.tensor(Xnum, dtype=torch.float)


# Converting the output to tensor
y = torch.tensor(data[outputs].values).flatten()


# Calculation of embedding sizes for the categorical values in the format (unique categorical values, embedding size (dimension of encoding))
categorical_column_sizes = [len(data[column].cat.categories) for column in categoricals]
categorical_embedding_sizes = [(col_size, min(50, (col_size+1)//2)) for col_size in categorical_column_sizes]


# Train-test split
totalnumber = len(Xnum)
testnumber = int(totalnumber * 0.2)

Xcattrain = Xcat[:totalnumber - testnumber]
Xcattest = Xcat[totalnumber - testnumber:totalnumber]
Xnumtrain = Xnum[:totalnumber - testnumber]
Xnumtest = Xnum[totalnumber - testnumber:totalnumber]
ytrain = y[:totalnumber - testnumber]
ytest = y[totalnumber - testnumber:totalnumber]

# Define initialization and forward pass
class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i,e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

# Define and show the model
model = Model(categorical_embedding_sizes, 1, 11, [10,10], p=0.4)
print(model)

# Loss function and optimization
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training of the model
def train():
    epochs = 100
    aggregated_losses = []
    
    for i in range(epochs):
        i += 1
        y_pred = model(Xcattrain, Xnumtrain)
        single_loss = loss_function(y_pred, ytrain)
        aggregated_losses.append(single_loss)
    
        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    
        optimizer.zero_grad()
        single_loss.backward()
        optimizer.step()
    
    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
    
    
    
    # Plot the loss over epocs
    plt.plot(range(epochs), aggregated_losses)
    plt.ylabel('Loss')
    plt.xlabel('epoch');
    
    
    # Predict on the test set
    with torch.no_grad():
        y_val = model(Xcattest, Xnumtest)
        loss = loss_function(y_val, ytest)
    
    
    y_val = np.argmax(y_val, axis=1)
    
    
    print(confusion_matrix(ytest,y_val))
    print(classification_report(ytest,y_val))
    print(accuracy_score(ytest, y_val))
    return
#model = Model(categorical_embedding_sizes, 1, 11, [100,100,50], p=0.4) og 300 epocs
#[[   0    1    0    1    0    0    1    0    0    0    0]
# [   7 3004  201  198   33    2  193    0   14   12    1]
# [   1 1217  231  196   12    4  172    2   33   15    3]
# [   0  842  160  293   30    3  256    1   43   21    1]
# [   2  481   87  158   27    6  219    1   47   11    2]
# [   0  414   54  137   29    4  250    1   45   15    0]
# [   0  253   24   68    9    0  449    3   55   15    0]
# [   0  202   14   49    1    2  352    2   42   19    2]
# [   0  130   14   51    5    2  196    3  104   65    7]
# [   0   97   17   53    4    2  118    1  101   69    7]
# [   1   84   10   31    4    1   89    2   55   78   12]]
#              precision    recall  f1-score   support
#
#           0       0.00      0.00      0.00         3
#           1       0.45      0.82      0.58      3665
#           2       0.28      0.12      0.17      1886
#           3       0.24      0.18      0.20      1650
#           4       0.18      0.03      0.05      1041
#           5       0.15      0.00      0.01       949
#           6       0.20      0.51      0.28       876
#           7       0.12      0.00      0.01       685
#           8       0.19      0.18      0.19       577
#           9       0.22      0.15      0.17       469
#          10       0.34      0.03      0.06       367
#
#    accuracy                           0.34     12168
#   macro avg       0.22      0.18      0.16     12168
#weighted avg       0.29      0.34      0.27     12168
#
#0.3447567389875082


#model = Model(categorical_embedding_sizes, 1, 11, [200,100,50,25], p=0.4) med 500 epocs og relu6
#[[   0    0    0    2    0    0    1    0    0    0    0]
# [   1 3024  194  192   33    4  161    0   34   15    7]
# [   1 1194  243  186   10    0  167    2   47   28    8]
# [   0  867  169  234   13    0  225    1   98   32   11]
# [   0  502   87  140   19    0  180    2   83   19    9]
# [   0  444   66  123   19    0  182    1   93   18    3]
# [   0  241   30   52    0    1  409    1  114   22    6]
# [   0  162   11   42    0    1  321    1  119   23    5]
# [   1  103   12   31    0    1   68    2  236  101   22]
# [   0   83    4   32    0    0   55    0  177   85   33]
# [   0   56    5   14    0    0   38    0  132   99   23]]
#              precision    recall  f1-score   support
#
#           0       0.00      0.00      0.00         3
#           1       0.45      0.83      0.58      3665
#           2       0.30      0.13      0.18      1886
#           3       0.22      0.14      0.17      1650
#           4       0.20      0.02      0.03      1041
#           5       0.00      0.00      0.00       949
#           6       0.23      0.47      0.30       876
#           7       0.10      0.00      0.00       685
#           8       0.21      0.41      0.28       577
#           9       0.19      0.18      0.19       469
#          10       0.18      0.06      0.09       367
#
#    accuracy                           0.35     12168
#   macro avg       0.19      0.20      0.17     12168
#weighted avg       0.27      0.35      0.28     12168
#
#0.3512491781722551