st.set_page_config(page_title="Income Prediction (Adult Census)", page_icon="💼")
st.title("💼 Income Prediction (Adult Census)")
st.caption("Perkiraan apakah penghasilan >50K USD/tahun berdasarkan data demografis.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("income_model.pkl")

model = load_model()

st.subheader("Input Data")

# Opsi kategori (disesuaikan dengan dataset Adult)
workclass_opts = [
    "Private","Self-emp-not-inc","Self-emp-inc","Federal-gov",
    "Local-gov","State-gov","Without-pay","Never-worked"
]
education_opts = [
    "Bachelors","Some-college","11th","HS-grad","Prof-school",
    "Assoc-acdm","Assoc-voc","9th","7th-8th","12th",
    "Masters","1st-4th","10th","Doctorate","5th-6th","Preschool"
]
marital_opts = [
    "Married-civ-spouse","Divorced","Never-married","Separated",
    "Widowed","Married-spouse-absent","Married-AF-spouse"
]
occupation_opts = [
    "Tech-support","Craft-repair","Other-service","Sales","Exec-managerial",
    "Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical",
    "Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv",
    "Armed-Forces"
]
relationship_opts = ["Wife","Own-child","Husband","Not-in-family","Other-relative","Unmarried"]
race_opts = ["White","Asian-Pac-Islander","Amer-Indian-Eskimo","Other","Black"]
gender_opts = ["Male","Female"]
country_opts = [
    "United-States","Cambodia","England","Puerto-Rico","Canada","Germany",
    "Outlying-US(Guam-USVI-etc)","India","Japan","Greece","South","China",
    "Cuba","Iran","Honduras","Philippines","Italy","Poland","Jamaica",
    "Vietnam","Mexico","Portugal","Ireland","France","Dominican-Republic",
    "Laos","Ecuador","Taiwan","Haiti","Columbia","Hungary","Guatemala",
    "Nicaragua","Scotland","Thailand","Yugoslavia","El-Salvador","Trinadad&Tobago",
    "Peru","Hong","Holand-Netherlands"
]

with st.form("form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 17, 90, 30)
        Workclass = st.selectbox("Workclass", workclass_opts)
        Final_Weight = st.number_input("Final Weight", 0, 1000000, 100000)
        Education = st.selectbox("Education", education_opts)
        EducationNum = st.number_input("EducationNum", 1, 16, 10)
        Marital_Status = st.selectbox("Marital Status", marital_opts)
        Occupation = st.selectbox("Occupation", occupation_opts)
    with col2:
        Relationship = st.selectbox("Relationship", relationship_opts)
        Race = st.selectbox("Race", race_opts)
        Gender = st.selectbox("Gender", gender_opts)
        Capital_Gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("capital loss", 0, 100000, 0)
        Hours_per_Week = st.number_input("Hours per Week", 1, 99, 40)
        Native_Country = st.selectbox("Native Country", country_opts)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Nama kolom harus EXACT sama dengan training
    input_df = pd.DataFrame([{
        "Age": Age,
        "Workclass": Workclass,
        "Final Weight": Final_Weight,
        "Education": Education,
        "EducationNum": EducationNum,
        "Marital Status": Marital_Status,
        "Occupation": Occupation,
        "Relationship": Relationship,
        "Race": Race,
        "Gender": Gender,
        "Capital Gain": Capital_Gain,
        "capital loss": capital_loss,
        "Hours per Week": Hours_per_Week,
        "Native Country": Native_Country
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Cari index proba kelas '>50K'
    classes = model.named_steps["classifier"].classes_
    idx_gt = np.where(classes == ">50K")[0][0]
    p_gt50k = float(proba[idx_gt])

    st.success(f"Prediksi: **{pred}**  |  Prob(>50K): **{p_gt50k:.3f}**")
    st.caption("Model: RandomForest pipeline dengan OneHotEncoder + StandardScaler + Imputer.")
