import streamlit as st
import requests

api_url = 'facerecognizeapi.azurewebsites.net/FACERECOGNIZE

def make_api_request(file1_path, file2_path):
    try:
        files = {
            'file1': open(file1_path, 'rb'),
            'file2': open(file2_path, 'rb')
        }

        response = requests.post(api_url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to connect to the API. Status code: {response.status_code}"}

    except Exception as e:
        return {"error": f"Error occurred: {str(e)}"}

def main():
    st.title("Image Comparison with API")
    st.write("Upload two images and compare them using the API.")

    uploaded_file1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

    if st.button("Compare Images") and uploaded_file1 and uploaded_file2:
        file1_path = './temp_image1.jpg'
        file2_path = './temp_image2.jpg'

        with open(file1_path, 'wb') as f:
            f.write(uploaded_file1.read())
        
        with open(file2_path, 'wb') as f:
            f.write(uploaded_file2.read())

        result = make_api_request(file1_path, file2_path)

        st.write("Comparison Result:")
        st.json(result)

        os.remove(file1_path)
        os.remove(file2_path)

if __name__ == "__main__":
    main()
