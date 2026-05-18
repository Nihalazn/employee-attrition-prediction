import pickle
import streamlit as st
from PIL import Image

model= pickle.load(open('attrition_prediction.save','rb'))
scaler=pickle.load(open('attrition_scaler.save','rb'))
le=pickle.load(open('attrition_encoder.save', 'rb'))
le1=pickle.load(open('attrition_encoder1.save', 'rb'))
le2=pickle.load(open('attrition_encoder2.save', 'rb'))
le3=pickle.load(open('attrition_encoder3.save', 'rb'))
le4=pickle.load(open('attrition_encoder4.save', 'rb'))

 # Session state for page
if "page" not in st.session_state:
    st.session_state.page = "home"
 
# ── FRONT PAGE ──
def home():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>📊 HR Attrition Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:rgba(255,255,255,0.6)!important;'>Predict whether an employee is likely to leave using Machine Learning</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Get Started →", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()

# ── PREDICTION PAGE ──
def main():
    if st.button("← Back"):
        st.session_state.page = "home"
        st.rerun()
 
    st.title(':green[HR ATTRITION]')
    img = Image.open('img.jpeg')
    st.image(img, width=600)
 
    age = st.number_input('Age', min_value=18, max_value=100)
    gen = st.radio('Gender', ['Male', 'Female'])
    mss = st.radio('Marital Status', ['Single', 'Married', 'Divorced'])
    jb  = st.radio('Job Level', ['1', '2', '3', '4', '5'])
    btd = st.radio('Business Travel Frequency', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    ovj = st.radio('Over Time', ['Yes', 'No'])
    mn  = st.number_input("Monthly Income", min_value=0, step=1)
    mr  = st.number_input("Monthly Rate", min_value=0, step=1)
    nw  = st.number_input("Number of Companies Worked", min_value=0)
    dh  = st.number_input("Distance From Home", min_value=0)
    ps  = st.number_input("Percent Salary Hike", min_value=0, max_value=100)
 
    pred = st.button('PREDICT')
 
    if pred:
        bt = le1.transform([btd])[0]
        ge = le2.transform([gen])[0]
        ms = le3.transform([mss])[0]
        ov = le4.transform([ovj])[0]
 
        features = [[int(age), bt, int(dh), ge, float(jb), ms, float(mn), float(mr), float(nw), ov, float(ps)]]
        scaled     = scaler.transform(features)
        prediction = model.predict(scaled)
        proba      = model.predict_proba(scaled)[0][1]  # probability of attrition
        risk_pct   = int(proba * 100)


        st.markdown("---")
        st.subheader("📋 Prediction Result")
 
        # Risk bar
        st.markdown(f"**Attrition Risk Score: {risk_pct}%**")
        st.progress(risk_pct)
 
        # Result + suggestion
        if prediction[0] == 1:
            st.error("⚠️ This employee is likely to leave.")
            if risk_pct >= 75:
                st.warning("🔴 High Risk — Schedule an immediate retention meeting. Review compensation and workload.")
            else:
                st.warning("🟡 Moderate-High Risk — Consider a check-in conversation and career growth opportunities.")
        else:
            st.success("✅ This employee is likely to stay.")
            if risk_pct <= 25:
                st.info("🟢 Low Risk — Employee appears satisfied. Keep up regular engagement.")
            else:
                st.info("🟡 Low-Moderate Risk — Generally stable, but worth a periodic check-in.")

 
# ── Router ──
if st.session_state.page == "home":
    home()
else:
    main()