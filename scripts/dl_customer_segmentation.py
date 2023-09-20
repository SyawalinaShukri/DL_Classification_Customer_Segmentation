#%%
#1. Import packages
import datetime
import pickle, os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import category_encoders as ce
import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


#%%
#2.Data Loading
PATH = os.getcwd()
csv_path = os.path.join(PATH,'train.csv')
df = pd.read_csv(csv_path)

# %%
#3. Data Inspection
print("Shape of the date =", df.shape)
print("\nInfo about the dataframe =\n", df.info())
print("\nDesc of the dataframe =\n", df.describe().transpose())
print("\nExample data =\n", df.head(1))

# %%
#4. Data Cleaning (EDA)
print(df.isna().sum())
print(df.duplicated().sum())

# %%
#check number of categories in each column
df.nunique()

#%%
df.columns  #get the keys of each col

#%%
#continuous data
df_con = df[['customer_age', 'balance','day_of_month','last_contact_duration',
       'num_contacts_in_campaign', 'days_since_prev_campaign_contact',
       'num_contacts_prev_campaign']]
#categorical data
df_cat = df.drop(df_con.columns, axis=1)
df_cat.drop("id", axis=1, inplace=True)

# %%
# 4.1 Setting Color Palette for data visualization

sns.palplot(sns.color_palette("Accent"))
sns.set_palette("Accent")
sns.set_style('whitegrid')

# %%
# 4.2 Histograms for all Continuous Variables

df_con.hist(bins=50, layout=(3,3))
plt.tight_layout()

# %%
# 4.3 Countplot Vs term_deposit_subscribed for all Categorical Variables
df_cat.drop("month", axis=1, inplace=True)
for column in df_cat.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=column, data=df_cat, hue='term_deposit_subscribed')

    plt.title(f'{column} vs Subscription')

# %%
# 4.4 countplot for all cat cols
fig, axs = plt.subplots(3,3, figsize=(6,6))

axs = axs.ravel()

for i, column in enumerate(df_cat.columns):
    sns.countplot(x=column, data=df_cat, ax=axs[i])

    axs[i].set_title(f'{column}')

plt.tight_layout()

# %%
# 4.5 Correlation Matrix
sns.heatmap(df_con.corr(), annot=True, cmap='coolwarm', linewidths=0.5)

# %%
#5. Data Cleaning
# 5.1 Dealing with missing values
#filter out missing values
df[df.isnull().any(axis=1)]

# %%
# (a) customer_age
#try 5 different ways of impute the missing values, and plot histogram & violin plots to see the differences#Create deep copies of the original dataframe fo each of the imputation methods
df_drop = df.dropna(subset=['customer_age']).copy()
df_zero = df.copy()
df_mode = df.copy()
df_mean = df.copy()
df_median = df.copy()

#Apply different imputation to each
df_zero['customer_age'].fillna(0, inplace=True)
df_mode['customer_age'].fillna(df['customer_age'].mode()[0], inplace=True)
df_mean['customer_age'].fillna(df['customer_age'].mean(), inplace=True)
df_median['customer_age'].fillna(df['customer_age'].median(), inplace=True)

#%%
#plot histograms and violins plots for each of the imputation methods
dfs = [df, df_drop, df_zero, df_mode, df_mean, df_median]
titles = ['Original', 'Drop NA', 'Fill with Zero', 'Fill with Mode', 'Fill with Mean', 'Fill with Median']
for i, (df_temp, title) in enumerate(zip(dfs, titles)):
    plt.figure(figsize=(6, 4))
    #hist
    plt.subplot(2,1,1)
    sns.histplot(df_temp['customer_age'], bins=100, color='cornflowerblue', edgecolor='k', kde=True)
    plt.title(f'Histogram of {title} customer_age')

    #violin
    plt.subplot(2,1,2)
    sns.violinplot(x = df_temp['customer_age'], color='lightsalmon')
    plt.title(f'Violin plot of {title} customer_age')
    plt.tight_layout()
    plt.show()

#%%
#print all the stats of Age col of the imputed dataframes
for df_temp, title in zip(dfs, titles):

    print(f'Descriptive statistics for {title} Age:')
    print(df_temp['customer_age'].describe())


#%%
#for the customer_age above, filling the null value with mean is ideal however there is also marital status that needs to be considered to be more precise and realistic
# (b) fill null value for marital with unknown as we can't assume whether their status
df["marital"] = df["marital"].fillna("unknown")
print(df['marital'].nunique())

#%%
#check the substantial differences to decide whether it is relevant or not to consider the marital status

# Group the data by marital status
age_by_marital = df.groupby('marital')['customer_age']

# Calculate summary statistics within each group
age_summary = age_by_marital.describe()

# Create a boxplot to visualize the age distribution within each marital status
plt.figure(figsize=(10, 6))
df.boxplot(column='customer_age', by='marital', figsize=(10, 6))


# Print the summary statistics
print(age_summary)

#%%
#since there are substantial differences bewteen customer_age and marital status, to fill the null age by based on their marital status can be considered
# Create a copy of the original DataFrame to store imputed ages
df_age = df.copy()

# Calculate the mean ages for each marital status category
mar = df_age[df_age["marital"] == "married"]["customer_age"].mean()
unm = df_age[df_age["marital"] == "single"]["customer_age"].mean()
unk = df_age[df_age["marital"] == "unknown"]["customer_age"].mean()

print(mar, unm, unk )

# Loop through the DataFrame to impute missing ages
for i in range(len(df_age)):
    if np.isnan(df_age["customer_age"][i]):
        if df_age["marital"][i] == "single":
            df_age.at[i, "customer_age"] = round(unm)
        if df_age["marital"][i] == "married":
            df_age.at[i, "customer_age"] = round(mar)
        if df_age["marital"][i] == "divorced":
            df_age.at[i, "customer_age"] = round(mar)
        if df_age["marital"][i] == "unknown":
            df_age.at[i, "customer_age"] = round(unk)

#%%
#plot histograms and violins plots for df_mean and df_age of the imputation methods
dfs = [df, df_age, df_mean]
titles = ['Original', 'Based on Marital', 'Fill with Mean']
for i, (df_temp, title) in enumerate(zip(dfs, titles)):
    plt.figure(figsize=(6, 4))
    #hist
    plt.subplot(2,1,1)
    sns.histplot(df_temp['customer_age'], bins=100, color='cornflowerblue', edgecolor='k', kde=True)
    plt.title(f'Histogram of {title} customer_age')

    #violin
    plt.subplot(2,1,2)
    sns.violinplot(x = df_temp['customer_age'], color='lightsalmon')
    plt.title(f'Violin plot of {title} customer_age')
    plt.tight_layout()
    plt.show()

#%%
#print all the stats of Age col of the imputed dataframes
for df_temp, title in zip(dfs, titles):

    print(f'Descriptive statistics for {title} Age:')
    print(df_temp['customer_age'].describe())

#There is not much difference since both based on mean, however fill with mean shows less outliers
#fill null age with mean
df['customer_age'] = df['customer_age'].fillna(df['customer_age'].mean())

print(df.isnull().sum())
print(df_con.columns)

#%%
#(c) Encoding: convert categorical data into a binary format

def one_hot_encoding(df,col):
    one_hot_encoder=ce.OneHotEncoder(cols=col,return_df=True,use_cat_names=True)
    df_final = one_hot_encoder.fit_transform(df)
    return df_final

print(df["term_deposit_subscribed"].value_counts())
df

#%%
#(d) fill othe null values for both categorical and numeric data
for i in df.columns:
    if i == "term_deposit_subscribed":
        continue
    if df[i].dtypes == "object" and i !="id":
        # print(i)
        df[i].fillna("unknown",inplace=True)
        df = one_hot_encoding(df,i)
        # print(len(df_train[i].value_counts()))
    elif df[i].isnull().sum()>0:
        if i == "num_contacts_in_campaign":
            df[i].fillna(1.0,inplace=True)
        else:
            df[i].fillna(round(df[i].mean()),inplace=True)

#%%
#(e)drop column id
df = df.drop("id", axis=1)

#inspect latest df
print(df.isna().sum())
print(df.duplicated().sum())

#%%
#5.2 Dealing with outliers
print(df.describe().transpose())
#plot a boxplot to see the outliers
# Calculate the number of rows and columns for the subplots
num_columns =  df.shape[1]
num_rows = (num_columns - 1) // 3 + 1

# Create a figure and a grid of subplots
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

# Flatten the axes array if it's not already flat
if num_rows > 1:
    axes = axes.flatten()

# Loop through the columns and create a boxplot for each
for i, col in enumerate(df.columns):
    ax = axes[i]
    df.boxplot(column=col, ax=ax)
    ax.set_title(col)  # Set the title to the column name

# Remove any empty subplots if num_columns is not a multiple of 3
for i in range(num_columns, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%%
# 6. Data Splitting
# Split the data into training and testing sets
X = df.drop('term_deposit_subscribed', axis=1)  # Features
y = df['term_deposit_subscribed']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Deal with class imbalance using oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

#%%
# 7. Data Normalization
# Standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

#%%
# 8.Model Development
# Define the deep learning model
# Define a learning rate scheduler
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay = 0.96
    lr = initial_learning_rate * (decay ** epoch)
    return lr

# Create the model with increased L2 regularization
def create_model(input_dim):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.7),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.7),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile the model
model = create_model(input_dim=X_train_resampled.shape[1])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement learning rate scheduling
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Create TensorBoard callback
base_log_path = r"logs"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#%%
# 9. Model Training
# Train the model with early stopping, learning rate scheduling, and increased L2 regularization
model.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=64, 
          validation_split=0.2, callbacks=[tb, early_stopping, lr_scheduler])
#%%
# 10. Model Evaluation
# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Test Accuracy: {accuracy}")

#%%
# 11. Hypertuning if needed
#since the accuracy is more than 70%, hypertuning won't be needed

#%%
#12. Save model in .h5 file
model.save('models/asessment3_model.h5')

# %%
#13. Plot and Save model architecture using plot_model function
def visualize_model(model, model_name, save_path):
    """
    Visualize and save the model architecture as a .png file.

    Args:
        model: The trained Keras model.
        model_name: A string representing the model name (used for the filename).
        save_path: The directory where the .png file will be saved.
    """
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Plot the model architecture
    plot_model(model, to_file=f"{save_path}/{model_name}_architecture.png", show_shapes=True, show_layer_names=True)

# Load your trained model here
trained_model = load_model('models/asessment3_model.h5')

# Call the function to visualize and save the model architecture
visualize_model(trained_model, "my_model", "model_visualization")
# %%
# %%
#14. Save important components so that we can deploy the NLP model elsewhere
#(A) scaler
scaler_save_path = os.path.join(PATH,"scaler.pkl")
with open(scaler_save_path, 'wb') as f:
    pickle.dump(scaler,f)
# %%
#Check if the scaler can be loaded 
with open(scaler_save_path, 'rb') as f:
    scaler_loaded = pickle.load(f)
print(type(scaler_loaded))

#%%
#(B) oversampler
oversampler_save_path = os.path.join(PATH,"oversampler.pkl")
with open(oversampler_save_path, 'wb') as f:
    pickle.dump(oversampler,f)
# %%
#Check if the oversampler can be loaded 
with open(oversampler_save_path, 'rb') as f:
    oversampler_loaded = pickle.load(f)
print(type(oversampler_loaded))
# %%
#(C) model
model_save_path = os.path.join(PATH, "campaign_outcome_model")
keras.models.save_model(model,model_save_path)
# %%
#Check if the model can be loaded
model_loaded = keras.models.load_model(model_save_path)
model_loaded.summary()
# %%