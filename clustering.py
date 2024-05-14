#Authored by Julius Välja and Rasmus Moorits Veski collectively

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

data_file_name = "full_results.csv"
dataset = pd.read_csv(data_file_name)

df_values = dataset.drop(columns=['submitdate. Date submitted',
       'lastpage. Last page',
       'startlanguage. Start language',
       'seed. Seed',
       'refurl. Referrer URL',
       'G33Q250. Please enter your Prolific ID:',
       'G02Q251. By starting the survey, you consent to participate in the research described above and allow the use of the anonymous data for educational and research purposes. Participation in the study is entirely voluntary. If for any reason you no longer wish to participate in the study, you may exit the survey before submitting the responses.',
       'G04Q11. How old are you?     ',
       'G02Q18. Please enter your citizenship:',
       'G04Q12. What is your highest completed level of education?',
       'G04Q13. What is your level of English proficiency?',
       'G04Q14. Do you have any previous experience in the field of machine learning?',
       'G04Q16. Do you have any previous experience with counterfactual explanation frameworks or causality frameworks?',
       'G02Q17. Do you have a medical background?',
       'G02Q08. From 1 (not at all) to 6 (perfectly), how well did you understand the metrics:'])

ids = df_values["id. Response ID"]
X = df_values.drop(columns=["Mean","id. Response ID"])
y = df_values["Mean"]
X_std = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=2,perplexity=2)
X_tsne = tsne.fit_transform(X_std)
X_tsne_data = np.vstack((X_tsne.T, y)).T

df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'Mean'])

clustering = DBSCAN(eps=9, min_samples=7).fit(df_tsne)

plt.scatter(df_tsne.Dim1, df_tsne.Dim2, s=80, c=clustering.labels_)
plt.savefig('tsne_dbscan.png')