from flask import Flask, render_template, request
from flask import  redirect, url_for
from PIL import Image
import werkzeug
from tensorflow.keras.preprocessing.image import  img_to_array
import cv2
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer
import joblib
import os
import  numpy as np
import pickle
from tensorflow import keras

# from keras.applications import ResNet50
# import sys
# import glob
# import re

# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

app= Flask(__name__)



# resnet = ResNet50(weights = 'imgagenet' , input_shape=(224,224,3) , pooling = 'avg')
# print("model loaded")
# MODEL_PATH = 'models/brain_tumor.h5'

# model = load_model(MODEL_PATH)
# model._make_predict_function()

# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds

# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)

#         # Process your result for human
#         # pred_class = preds.argmax(axis=-1)            # Simple argmax
#         pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
#         result = str(pred_class[0][0][1])               # Convert to string
#         return result
#     return None

# -----------------------------------------------------------------------------------------



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home_stroke")
def home_stroke():
    return render_template("home_stroke.html")

@app.route("/home_heart")
def home_heart():
    return render_template("home_heart.html")

@app.route("/home_diabetes")
def home_diabetes():
    return render_template("home_diabetes.html")

@app.route("/home_kidney")
def home_kidney():
    return render_template("home_kidney.html")

@app.route("/home_liver")
def home_liver():
    return render_template("home_liver.html")

@app.route("/home_malaria")
def home_malaria():
    return render_template("home_malaria.html")

@app.route("/home_hepatitis")
def home_hepatitis():
    return render_template("home_hepatitis.html")

@app.route("/home_brain_tumor")
def home_brain_tumor():
    return render_template("home_brain_tumor.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/brain_tumor_pred",methods=['POST','GET'])
def brain_tumor_pred():
    img = request.files['img']
    img.save('uploads/brain_tumor_img.jpg')

    image = Image.open("uploads/brain_tumor_img.jpg")
    model = keras.models.load_model('models/brain_tumor.h5')
    # image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image ,(224,224),interpolation = cv2.INTER_AREA)
    # image = img_to_array(image)
    # image = np.array([i[0] for i in image]).reshape(-1,128,128,1)
    # image = np.reshape(image, (1,128,128,1))
    # image = np.array(image.resize((128,128)))
    x = np.array(image.resize((128,128)))
    # x = cv2.resize(image,(128,128))
    x = x.reshape(1,128,128,3)
    res = model.predict_on_batch(x)
    pred = np.where(res == np.amax(res))[1][0]
    
    # pred = model.predict_on_batch(image)

    # pred = np.argmax(pred)
    # pred = resnet.pred(image)

    # pred = np.argmax(pred)
    if pred==1:
        return render_template('no_brain_tumor.html')
    else:
        return render_template('brain_tumor_yes.html')


@app.route("/heart_pred",methods=['POST','GET'])
def heart_pred():
    age=int(request.form['age'])
    cholesterol=int(request.form['cholesterol'])
    fasting_blood_sugar=int(request.form['fasting_blood_sugar'])
    max_heart_rate_achieved = int(request.form['max_heart_rate_achieved'])
    exercise_induced_angina = int(request.form['exercise_induced_angina'])
    st_depression = float(request.form['st_depression'])
    chest_pain_type_typical_angina = int(request.form['chest_pain_type_typical angina'])
    rest_ecg_left_ventricular_hypertrophy = float(request.form['rest_ecg_left ventricular hypertrophy'])
    rest_ecg_normal = float(request.form['rest_ecg_normal'])
    st_slope_flat = int(request.form['st_slope_flat'])
    st_slope_upsloping = int(request.form['st_slope_upsloping'])

    x=np.array([age,cholesterol,fasting_blood_sugar,max_heart_rate_achieved,exercise_induced_angina,st_depression,
                chest_pain_type_typical_angina,rest_ecg_left_ventricular_hypertrophy,rest_ecg_normal,st_slope_flat,st_slope_upsloping]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_heart.pkl')
    scaler_heart=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_heart=pickle.load(scaler_file)

    x=scaler_heart.transform(x)

    model_path=os.path.join('models/rf_ent_heart.sav')
    rf_ent=joblib.load(model_path)

    Y_pred=rf_ent.predict(x)

    # for No Heart Risk
    if Y_pred==0:
        return render_template('no_heart_disease.html')
    else:
        return render_template('heart_disease.html')

#stroke

@app.route("/stroke_pred",methods=['POST','GET'])
def stroke_pred():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models/dt.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

# diabetes
@app.route("/diabetes_pred",methods=['POST','GET'])
def diabetes_pred():
    
    Polyuria=int(request.form['Polyuria'])
    Polydipsia = int(request.form['Polydipsia'])
    age=int(request.form['age'])
    Gender=int(request.form['Gender'])
    partial_paresis	 = int(request.form['partial paresis'])
    sudden_wieght_loss = int(request.form['sudden wieght loss'])
    Irritability = int(request.form['Irritability'])
    delayed_healing	 = int(request.form['delayed healing'])
    Alopecia = int(request.form['Alopecia'])
    Itching = int(request.form['Itching'])
    
    x=np.array([Polyuria,Polydipsia,age,Gender,partial_paresis,sudden_wieght_loss,Irritability,delayed_healing,Alopecia,Itching]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_diabetes.pkl')
    scaler_diabetes=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_diabetes=pickle.load(scaler_file)

    x=scaler_diabetes.transform(x)

    model_path=os.path.join('models/rf_diabetes.sav')
    rf_diabetes=joblib.load(model_path)

    Y_pred=rf_diabetes.predict(x)

    # for No Diabetes Risk
    if Y_pred==0:
        return render_template('no_diabetes.html')
    else:
        return render_template('diabetes_yes.html')

#kidney

@app.route("/kidney_pred",methods=['POST','GET'])
def kidney_pred():
    sg=float(request.form['sg'])
    al=float(request.form['al'])
    sc=float(request.form['sc'])
    hemo = float(request.form['hemo'])
    pcv = int(request.form['pcv'])
    htn = int(request.form['htn'])

    x=np.array([sg,al,sc,hemo,pcv,htn]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_kidney.pkl')
    scaler_kidney=None
    with open(scaler_path,'rb') as scaler_file:
        scaler_kidney=pickle.load(scaler_file)

    x=scaler_kidney.transform(x)

    model_path=os.path.join('models/rf_kidney.sav')
    rf_kidney=joblib.load(model_path)

    Y_pred=rf_kidney.predict(x)

    # for No ckd Risk
    if Y_pred==0:
        return render_template('no_kidney_disease.html')
    else:
        return render_template('kidney_disease_yes.html')

#liver
@app.route("/liver_pred",methods=['POST','GET'])
def liver_pred():
    age=int(request.form['age'])
    gender=int(request.form['gender'])
    Total_Bilirubin	=float(request.form['Total_Bilirubin'])
    Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
    Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
    Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
    Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
    Total_Protiens = float(request.form['Total_Protiens'])
    Albumin = float(request.form['Albumin'])
    Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])

    x=np.array([age,gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_liver.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models/sv.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Liver Risk
    if Y_pred==0:
        return render_template('no_liver_disease.html')
    else:
        return render_template('liver_disease.html')

#malaria
@app.route("/malaria_pred",methods=['POST','GET'])
def malaria_pred():
    sex=int(request.form['sex'])
    fever=int(request.form['fever'])
    cold=int(request.form['cold'])
    rigor = int(request.form['rigor'])
    fatigue = int(request.form['fatigue'])
    headace = int(request.form['headace'])
    bitter_tongue = int(request.form['bitter_tongue'])
    vomitting = int(request.form['vomitting'])
    diarrhea = int(request.form['diarrhea'])
    Convulsion = int(request.form['Convulsion'])
    Anemia = int(request.form['Anemia'])
    jundice = int(request.form['jundice'])
    cocacola_urine = int(request.form['cocacola_urine'])
    hypoglycemia = int(request.form['hypoglycemia'])
    prostraction = int(request.form['prostraction'])
    hyperpyrexia = int(request.form['hyperpyrexia'])
    x=np.array([sex,fever,cold,rigor,fatigue,headace,bitter_tongue,vomitting,diarrhea,
               Convulsion,Anemia,jundice,cocacola_urine,hypoglycemia,prostraction,hyperpyrexia]).reshape(1,-1)

    scaler_path=os.path.join('models/scaler_malaria.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('models/gnb_malaria.sav')
    gnb_malaria=joblib.load(model_path)

    Y_pred=gnb_malaria.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('no_malaria.html')
    else:
        return render_template('malaria_disease.html')

#hepatitis
@app.route("/hepatitis_pred",methods=['POST','GET'])
def hepatitis_pred():
    age=int(request.form['age'])
    sex=int(request.form['sex'])
    steroid=int(request.form['steroid'])
    antivirals = int(request.form['antivirals'])
    fatigue = int(request.form['fatigue'])
    spiders = int(request.form['spiders'])
    ascites = int(request.form['ascites'])
    varices = int(request.form['varices'])
    bilirubin = float(request.form['bilirubin'])
    alk_phosphate = int(request.form['alk_phosphate'])
    sgot = int(request.form['sgot'])
    albumin = float(request.form['albumin'])
    protime = int(request.form['protime'])
    histology = int(request.form['histology'])
    x=np.array([age,sex,steroid,antivirals,fatigue,spiders,ascites,varices,bilirubin,
               alk_phosphate,sgot,albumin,protime,histology]).reshape(1,-1)

    # scaler_path=os.path.join('models/scaler_malaria.pkl')
    # scaler=None
    # with open(scaler_path,'rb') as scaler_file:
    #     scaler=pickle.load(scaler_file)

    # x=scaler.transform(x)

    model_path=os.path.join('models/rf_hepatitis.sav')
    rf_hepatitis=joblib.load(model_path)

    Y_pred=rf_hepatitis.predict(x)

    # for No Stroke Risk
    if Y_pred==2:
        return render_template('no_hepatitis.html')
    else:
        return render_template('hepatitis_yes.html')

#brain tumor
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404

if __name__=="__main__":
    app.run(debug=True,port=7384)