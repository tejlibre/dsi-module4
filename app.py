import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Snake Classifier')

st.markdown("Welcome to this simple web application that classifies snakes. The snakes are classified into 135 different classes.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "notebooks/saved_model/serialized_snake_classifier_model_not_tuned.h5"
    IMAGE_SHAPE = (180, 180,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((180,180))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [18, 20, 25, 26, 39, 41, 48, 52, 54, 57, 60, 65, 71, 73, 83, 87, 88, 99, 110, 111, 113, 114, 122, 135, 140, 155, 158, 159, 162, 163, 165, 168, 169, 175, 177, 180, 185, 186, 188, 189, 191, 193,
       195, 203, 215, 216, 220, 226, 238, 255, 263, 280, 284, 302, 315, 319, 321, 323, 335, 338, 345, 348, 352, 360, 363, 364, 368, 373, 381, 383, 384, 396, 422, 424, 427, 429, 430, 441, 448, 453,
       454, 457, 462, 464, 470, 474, 477, 481, 485, 495, 497, 507, 515, 544, 545, 546, 560, 562, 575, 576, 578, 590, 603, 605, 609, 616, 617, 619, 623, 628, 629, 634, 651, 652, 656, 672, 674, 675,
       678, 686, 690, 691, 696, 698, 699, 701, 725, 738, 740, 741, 746, 747, 751, 755, 758]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
        '18',: 0,
        '20',: 0,
        '25',: 0,
        '26',: 0,
        '39',: 0,
        '41',: 0,
        '48',: 0,
        '52',: 0,
        '54',: 0,
        '57',: 0,
        '60',: 0,
        '65',: 0,
        '71',: 0,
        '73',: 0,
        '83',: 0,
        '87',: 0,
        '88',: 0,
        '99',: 0,
        '110',: 0,
        '111',: 0,
        '113',: 0,
        '114',: 0,
        '122',: 0,
        '135',: 0,
        '140',: 0,
        '155',: 0,
        '158',: 0,
        '159',: 0,
        '162',: 0,
        '163',: 0,
        '165',: 0,
        '168',: 0,
        '169',: 0,
        '175',: 0,
        '177',: 0,
        '180',: 0,
        '185',: 0,
        '186',: 0,
        '188',: 0,
        '189',: 0,
        '191',: 0,
        '193',: 0,
        '195',: 0,
        '203',: 0,
        '215',: 0,
        '216',: 0,
        '220',: 0,
        '226',: 0,
        '238',: 0,
        '255',: 0,
        '263',: 0,
        '280',: 0,
        '284',: 0,
        '302',: 0,
        '315',: 0,
        '319',: 0,
        '321',: 0,
        '323',: 0,
        '335',: 0,
        '338',: 0,
        '345',: 0,
        '348',: 0,
        '352',: 0,
        '360',: 0,
        '363',: 0,
        '364',: 0,
        '368',: 0,
        '373',: 0,
        '381',: 0,
        '383',: 0,
        '384',: 0,
        '396',: 0,
        '422',: 0,
        '424',: 0,
        '427',: 0,
        '429',: 0,
        '430',: 0,
        '441',: 0,
        '448',: 0,
        '453',: 0,
        '454',: 0,
        '457',: 0,
        '462',: 0,
        '464',: 0,
        '470',: 0,
        '474',: 0,
        '477',: 0,
        '481',: 0,
        '485',: 0,
        '495',: 0,
        '497',: 0,
        '507',: 0,
        '515',: 0,
        '544',: 0,
        '545',: 0,
        '546',: 0,
        '560',: 0,
        '562',: 0,
        '575',: 0,
        '576',: 0,
        '578',: 0,
        '590',: 0,
        '603',: 0,
        '605',: 0,
        '609',: 0,
        '616',: 0,
        '617',: 0,
        '619',: 0,
        '623',: 0,
        '628',: 0,
        '629',: 0,
        '634',: 0,
        '651',: 0,
        '652',: 0,
        '656',: 0,
        '672',: 0,
        '674',: 0,
        '675',: 0,
        '678',: 0,
        '686',: 0,
        '690',: 0,
        '691',: 0,
        '696',: 0,
        '698',: 0,
        '699',: 0,
        '701',: 0,
        '725',: 0,
        '738',: 0,
        '740',: 0,
        '741',: 0,
        '746',: 0,
        '747',: 0,
        '751',: 0,
        '755',: 0,
        '758',: 0
    }

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result









    

if __name__ == "__main__":
    main()


