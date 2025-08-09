import streamlit as st
import pandas as pd
import joblib

# Impor kelas-kelas yang diperlukan dari scikit-learn
# Ini sangat penting karena model.pkl Anda menggunakan objek-objek ini
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Load model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Error: The model file 'best_model.pkl' was not found. Please make sure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

# Load data untuk mengisi opsi di sidebar
try:
    data = pd.read_csv('MEN_SHOES.csv')
    
    # Membersihkan dan mengubah tipe data
    # Menghapus '₹' dan ',' dari Current_Price dan mengubahnya menjadi float
    data['Current_Price'] = data['Current_Price'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '', regex=False).astype(float)
    # Menghapus ',' dari How_Many_Sold dan mengubahnya menjadi integer
    data['How_Many_Sold'] = data['How_Many_Sold'].astype(str).str.replace(',', '', regex=False).astype(float)

    brand_options = data['Brand_Name'].unique()
    product_options = data['Product_details'].unique()

except FileNotFoundError:
    st.error("Error: The data file 'MEN_SHOES.csv' was not found. Please make sure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the data: {e}")
    st.stop()

# Application title
st.title('Men Shoes Rating Prediction App')

# Sidebar untuk input pengguna
st.sidebar.header('User Input Parameters')

def user_input_features():
    brand_name = st.sidebar.selectbox('Brand Name', brand_options)
    # Catatan: Kolom Product_details tidak digunakan oleh model pipeline yang Anda latih,
    # tetapi tetap ditampilkan di UI untuk kelengkapan.
    product_details = st.sidebar.selectbox('Product Details', product_options)
    how_many_sold = st.sidebar.number_input('How Many Sold', value=int(data['How_Many_Sold'].mean()), min_value=0)
    current_price = st.sidebar.number_input('Current Price (IDR)', value=float(data['Current_Price'].mean()), min_value=0.0)
    
    data_dict = {
        'Brand_Name': brand_name,
        'How_Many_Sold': how_many_sold,
        'Current_Price': current_price,
        'Product_details': product_details
    }
    features = pd.DataFrame(data_dict, index=[0])
    return features

df = user_input_features()

# Menampilkan input pengguna
st.subheader('User Input parameters')
st.write(df)

# Membuat prediksi
if st.button('Predict'):
    try:
        # Pipeline secara otomatis akan menangani pra-pemrosesan
        prediction = model.predict(df)
        
        # Menampilkan prediksi
        st.subheader('Predicted Rating')
        st.write(f"The predicted rating is: **{prediction[0]:.2f}**")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")