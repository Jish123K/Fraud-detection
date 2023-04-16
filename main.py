import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Load the transaction data

df = pd.read_csv('transactions.csv')

# Convert the transaction data to a graph

G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.DiGraph())

# Calculate the degree centrality of each node in the graph

d = nx.degree_centrality(G)

# Plot the degree centrality of each node

plt.figure()

plt.scatter(d.values(), np.arange(len(d)), c=d.values(), cmap='Reds')

plt.xlabel('Degree Centrality')

plt.ylabel('Node ID')

plt.show()

# Identify the top 10 nodes with the highest degree centrality

top_nodes = np.argsort(d.values())[-10:]

# Plot the network of the top 10 nodes

plt.figure()

nx.draw(G, nodelist=top_nodes, node_color=d[top_nodes], node_size=d[top_nodes], edge_color='black')

plt.show()

# Calculate the clustering coefficient of each node in the graph

c = nx.clustering_coefficient(G)

# Plot the clustering coefficient of each node

plt.figure()

plt.scatter(c.values(), np.arange(len(c)), c=c.values(), cmap='Greens')

plt.xlabel('Clustering Coefficient')

plt.ylabel('Node ID')

plt.show()

# Identify the top 10 nodes with the highest clustering coefficient

top_nodes = np.argsort(c.values())[-10:]

# Plot the network of the top 10 nodes

plt.figure()

nx.draw(G, nodelist=top_nodes, node_color=c[top_nodes], node_size=c[top_nodes], edge_color='black')

plt.show()
# Calculate the betweenness centrality of each node in the graph

b = nx.betweenness_centrality(G)

# Plot the betweenness centrality of each node

plt.figure()

plt.scatter(b.values(), np.arange(len(b)), c=b.values(), cmap='Blues')

plt.xlabel('Betweenness Centrality')

plt.ylabel('Node ID')

plt.show()

# Identify the top 10 nodes with the highest betweenness centrality

top_nodes = np.argsort(b.values())[-10:]

# Plot the network of the top 10 nodes

plt.figure()

nx.draw(G, nodelist=top_nodes, node_color=b[top_nodes], node_size=b[top_nodes], edge_color='black')

plt.show()

# Calculate the PageRank of each node in the graph

p = nx.pagerank(G)

# Plot the PageRank of each node

plt.figure()

plt.scatter(p.values(), np.arange(len(p)), c=p.values(), cmap='Purples')

plt.xlabel('PageRank')

plt.ylabel('Node ID')

plt.show()

# Identify the top 10 nodes with the highest PageRank

top_nodes = np.argsort(p.values())[-10:]

# Plot the network of the top 10 nodes

plt.figure()

nx.draw(G, nodelist=top_nodes, node_color=p[top_nodes], node_size=p[top_nodes], edge_color='black')

plt.show()

# Train a GCN model on the transaction data

model = GraphSAGE(G, in_feats=1, out_feats=1, hidden_feats=[16, 8], num_layers=2, dropout=0.5)

model.fit(df)
# Calculate the accuracy of the model

acc = np.mean(preds == df['is_fraudulent'])

print('Accuracy:', acc)

# Plot the ROC curve of the model

fpr, tpr, thresholds = metrics.roc_curve(df['is_fraudulent'], preds)

auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.legend()

plt.show()

# Evaluate the model on a held-out test set

X_test, y_test = df_test.drop('is_fraudulent', axis=1), df_test['is_fraudulent']

preds_test = model.predict(X_test)

acc_test = np.mean(preds_test == y_test)

print('Accuracy on test set:', acc_test)

# Save the model

model.save('model.pkl')

# Load the model

model = GraphSAGE.load('model.pkl')

# Predict the probability of a new transaction being fraudulent

new_transaction = pd.DataFrame({'source': 1, 'target': 2, 'amount': 100})

pred = model.predict(new_transaction)

print('Probability of new transaction being fraudulent:', pred)
# Load the model

model = GraphSAGE.load('model.pkl')

# Predict the probability of a new transaction being fraudulent

new_transaction = pd.DataFrame({'source': 1, 'target': 2, 'amount': 100})

pred = model.predict(new_transaction)

print('Probability of new transaction being fraudulent:', pred)

# If the probability is greater than a certain threshold, flag the transaction as fraudulent.

threshold = 0.5

if pred > threshold:

    print('Transaction is fraudulent.')

else:

    print('Transaction is not fraudulent.')

# If the transaction is fraudulent, take appropriate action, such as blocking the transaction or contacting the customer.

if pred > threshold:

    # Block the transaction.

    print('Transaction blocked.')

    # Contact the customer.

    print('Contacting customer...')

# If the transaction is not fraudulent, allow the transaction to proceed.

else:

    print('Trans# If the transaction is not fraudulent, allow the transaction to proceed.

else:

    print('Transaction allowed to proceed.')

# Add a user interface to allow users to view the results of the fraud detection model.

import streamlit as st

# Create a sidebar to display the results of the fraud detection model.

st.sidebar.title('Fraud Detection Results')

st.sidebar.write('The probability of this transaction being fraudulent is:', pred)

# If the transaction is fraudulent, display a message to the user and ask them to contact customer service.

if pred > threshold:

    st.write('This transaction is fraudulent. Please contact customer service.')

# If the transaction is not fraudulent, display a message to the user and allow them to proceed with the transaction.

else:

    st.write('This transaction is not fraudulent. You may proceed.')

# Add a feature to allow users to view the graph of the transaction network.

import networkx as nx

# Create a graph of the transaction network.

G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.DiGraph())

# Display the graph of the transaction network.

st.graphviz(nx.drawing.nx_agraph.to_agraph(G).draw())

# Add a feature to allow users to view the details of a specific transaction.

def view_transaction(transaction_id):

    # Get the details of the transaction.

    transaction = df[df['id'] == transaction_id]

    # Display the details of the transaction.

    st.write('Transaction ID:', transaction['id'])

    st.write('Source Account:', transaction['source'])

    st.write('Target Account:', transaction['target'])

    st.write('Amount:', transaction['amount'])

    st.write('Date:', transaction['date'])

    st.write('Time:', transaction['time'])

# Allow users to view the details of a specific transaction.

st.write('View Transaction')

transaction_id = st.text_input('Enter the transaction ID:')

if transaction_id:
          view_transaction(transaction_id)
          # Add a feature to the model that can detect suspicious transactions.

import numpy as np

# Create a list of features that are indicative of suspicious transactions.

suspicious_features = ['amount', 'time', 'location', 'device', 'ip_address', 'browser', 'operating_system', 'payment_method', 'product_or_service']

# Calculate the z-score of each feature for each transaction.

z_scores = np.zeros((len(df), len(suspicious_features)))

for i in range(len(df)):

    for j in range(len(suspicious_features)):

        z_scores[i, j] = (df[suspicious_features[j]][i] - np.mean(df[suspicious_features[j]])) / np.std(df[suspicious_features[j]])

# Create a new feature that is the sum of the z-scores for all of the suspicious features.

suspicious_score = np.sum(z_scores, axis=1)

# Add the suspicious_score feature to the dataframe.

df['suspicious_score'] = suspicious_score

# Train a new model on the dataframe with the added suspicious_score feature.

model = GraphSAGE(G, in_feats=1, out_feats=1, hidden_feats=[16, 8], num_layers=2, dropout=0.5)

model.fit(df)

# Predict the probability of each transaction being fraudulent, using the new model.

preds = model.predict(df)

# If the probability is greater than a certain threshold, flag the transaction as fraudulent.

threshold = 0.5

if pred > threshold:

    print('Transaction is fraudulent.')

else:

    print('Transaction is not fraudulent.')

# If the transaction is fraudulent, take appropriate action, such as blocking the transaction or contacting the customer.

if pred > threshold:

    # Block the transaction.

    print('Transaction blocked.')

    # Contact the customer.

    print('Contacting customer...')
          # If the transaction is not fraudulent, allow the transaction to proceed.

else:

    print('Transaction allowed to proceed.')

# Add a feature to the model that can identify fraudulent networks of accounts.

import networkx as nx

# Create a graph of the transaction network.

G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.DiGraph())

# Identify fraudulent networks of accounts.

fraudulent_networks = nx.find_cliques(G, min_size=2)

# Print the fraudulent networks.

for fraudulent_network in fraudulent_networks:

    print(fraudulent_network)

# Add a feature to the model that can predict the amount of money that will be lost in a fraudulent transaction.

import numpy as np

# Calculate the mean amount of money lost in fraudulent transactions.

mean_loss = np.mean(df['amount'][df['is_fraudulent'] == True])

# Create a new feature that is the predicted amount of money that will be lost in each transaction.

predicted_loss = np.where(df['is_fraudulent'], mean_loss, 0)

# Add the predicted_loss feature to the dataframe.

df['predicted_loss'] = predicted_loss

# Add a feature to the model that can identify the type of fraud that is being committed.

import pandas as pd

# Create a list of different types of fraud.
          fraud_targets = ['individual', 'business', 'government']

# Create a new feature that is the target of the fraud.

fraud_target = pd.Categorical(df['target'], categories=fraud_targets)

# Add the fraud_target feature to the dataframe.

df['fraud_target'] = fraud_target

# Add a feature to the model that can identify the time of the fraud.

import numpy as np

# Create a list of different times of fraud.

fraud_times = ['day', 'night']

# Create a new feature that is the time of the fraud.

fraud_time = pd.Categorical(df['time'], categories=fraud_times)

# Add the fraud_time feature to the dataframe.

df['fraud_time'] = fraud_time

# Add a feature to the model that can identify the location of the fraud.

import numpy as np

# Create a list of different locations of fraud.

fraud_locations = ['home', 'work', 'public']

# Create a new feature that is the location of the fraud.

fraud_location = pd.Categorical(df['location'], categories=fraud_locations)

# Add the fraud_location feature to the dataframe.

df['fraud_location'] = fraud_location

# Add a feature to the model that can identify the device that was used to commit the fraud.

import numpy as np

# Create a list of different devices that were used to commit fraud.

fraud_devices = ['computer', 'phone', 'tablet']

# Create a new feature that is the device that was used to commit the fraud.

fraud_device = pd.Categorical(df['device'], categories=fraud_devices)

fraud_types = ['credit card fraud', 'identity theft', 'wire fraud', 'check fraud', 'ATM fraud']

# Create a new feature that is the type of fraud that is being committed.

fraud_type = pd.Categorical(df['type'], categories=fraud_types)

# Add the fraud_type feature to the dataframe.

df['fraud_type'] = fraud_type

# Add a feature to the model that can identify the source of the fraud.

import numpy as np
  # Create a new feature that is the IP address that was used to commit the fraud.

fraud_ip_address = df['ip_address']

# Add the fraud_ip_address feature to the dataframe.

df['fraud_ip_address'] = fraud_ip_address

# Create a new feature that is the browser that was used to commit the fraud.

fraud_browser = df['browser']

# Add the fraud_browser feature to the dataframe.

df['fraud_browser'] = fraud_browser

# Create a new feature that is the operating system that was used to commit the fraud.

fraud_operating_system = df['operating_system']

# Add the fraud_operating_system feature to the dataframe.

df['fraud_operating_system'] = fraud_operating_system

# Create a new feature that is the payment method that was used to commit the fraud.

fraud_payment_method = df['payment_method']

# Add the fraud_payment_method feature to the dataframe.

df['fraud_payment_method'] = fraud_payment_method

# Create a new feature that is the product or service that was purchased in the fraudulent transaction.

fraud_product_or_service = df['product_or_service']

# Add the fraud_product_or_service feature to the dataframe.

df['fraud_product_or_service'] = fraud_product_or_service

# Create a new feature that is the customer who committed the fraud.

fraud_customer = df['customer']

# Add the fraud_customer feature to the dataframe.

df['fraud_customer'] = fraud_customer

# Create a new feature that is the merchant who was the victim of the fraud.

fraud_merchant = df['merchant']

# Add the fraud_merchant feature to the dataframe.

df['fraud_merchant'] = fraud_merchant

# Create a new feature that is the bank that processed the fraudulent transaction.

fraud_bank = df['bank']
          # Add the fraud_bank feature to the dataframe.

df['fraud_bank'] = fraud_bank

# Create a new feature that is the credit card company that issued the card that was used in the fraudulent transaction.

fraud_credit_card_company = df['credit_card_company']

# Add the fraud_credit_card_company feature to the dataframe.

df['fraud_credit_card_company'] = fraud_credit_card_company

# Create a new feature that is the insurance company that will be liable for the fraudulent transaction.

fraud_insurance_company = df['insurance_company']

# Add the fraud_insurance_company feature to the dataframe.

df['fraud_insurance_company'] = fraud_insurance_company

# Train a new model on the dataframe with all of the added features.

model = GraphSAGE(G, in_feats=1, out_feats=1, hidden_feats=[16, 8], num_layers=2, dropout=0.5)

model.fit(df)

# Predict the probability of each transaction being fraudulent, using the new model.

preds = model.predict(df)

# If the probability is greater than a certain threshold, flag the transaction as fraudulent.

threshold = 0.5

if pred > threshold:

    print('Transaction is fraudulent.')

else:

    print('Transaction is not fraudulent.')

# If the transaction is fraudulent, take appropriate action, such as blocking the transaction or contacting the customer.

if pred > threshold:

    # Block the transaction.

    print('Transaction blocked.')

    # Contact the customer.

    print('Contacting customer...')

# If the transaction is not fraudulent, allow the transaction to proceed.

else:

    print('Transaction allowed to proceed.')
          # End the program.

print('Thank you for using our fraud detection service.')
    
