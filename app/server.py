from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from fastapi.templating import Jinja2Templates
from fastapi import Request

# Load the trained model
model = joblib.load('app/model.joblib')

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Example label encoders for categorical variables (you'll need to adjust this to match your training)
marital_status_encoder = LabelEncoder()
house_ownership_encoder = LabelEncoder()
car_ownership_encoder = LabelEncoder()
profession_encoder = LabelEncoder()
city_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Example of fitting encoders with training data categories (replace with your actual categories)
marital_status_encoder.fit(['single', 'married'])
house_ownership_encoder.fit(['rented', 'norent_noown', 'owned'])
car_ownership_encoder.fit(['no', 'yes'])
profession_encoder.fit(['Mechanical_engineer', 'Software_Developer', 'Technical_writer',
       'Civil_servant', 'Librarian', 'Economist', 'Flight_attendant',
       'Architect', 'Designer', 'Physician', 'Financial_Analyst',
       'Air_traffic_controller', 'Politician', 'Police_officer', 'Artist',
       'Surveyor', 'Design_Engineer', 'Chemical_engineer',
       'Hotel_Manager', 'Dentist', 'Comedian', 'Biomedical_Engineer',
       'Graphic_Designer', 'Computer_hardware_engineer',
       'Petroleum_Engineer', 'Secretary', 'Computer_operator',
       'Chartered_Accountant', 'Technician', 'Microbiologist',
       'Fashion_Designer', 'Aviator', 'Psychologist', 'Magistrate',
       'Lawyer', 'Firefighter', 'Engineer', 'Official', 'Analyst',
       'Geologist', 'Drafter', 'Statistician', 'Web_designer',
       'Consultant', 'Chef', 'Army_officer', 'Surgeon', 'Scientist',
       'Civil_engineer', 'Industrial_Engineer', 'Technology_specialist'])
city_encoder.fit(['Rewa', 'Parbhani', 'Alappuzha', 'Bhubaneswar', 'Tiruchirappalli',
       'Jalgaon', 'Tiruppur', 'Jamnagar', 'Kota', 'Karimnagar', 'Hajipur',
       'Adoni', 'Erode', 'Kollam', 'Madurai', 'Anantapuram', 'Kamarhati',
       'Bhusawal', 'Sirsa', 'Amaravati', 'Secunderabad', 'Ahmedabad',
       'Ajmer', 'Ongole', 'Miryalaguda', 'Ambattur', 'Indore',
       'Pondicherry', 'Shimoga', 'Chennai', 'Gulbarga', 'Khammam',
       'Saharanpur', 'Gopalpur', 'Amravati', 'Udupi', 'Howrah',
       'Aurangabad', 'Hospet', 'Shimla', 'Khandwa', 'Bidhannagar',
       'Bellary', 'Danapur', 'Purnia', 'Bijapur', 'Patiala', 'Malda',
       'Sagar', 'Durgapur', 'Junagadh', 'Singrauli', 'Agartala',
       'Thanjavur', 'Hindupur', 'Naihati', 'North_Dumdum', 'Panchkula',
       'Anantapur', 'Serampore', 'Bathinda', 'Nadiad', 'Kanpur',
       'Haridwar', 'Berhampur', 'Jamshedpur', 'Hyderabad', 'Bidar',
       'Kottayam', 'Solapur', 'Suryapet', 'Aizawl', 'Asansol', 'Deoghar',
       'Eluru', 'Ulhasnagar', 'Aligarh', 'South_Dumdum', 'Berhampore',
       'Gandhinagar', 'Sonipat', 'Muzaffarpur', 'Raichur',
       'Rajpur_Sonarpur', 'Ambarnath', 'Katihar', 'Kozhikode', 'Vellore',
       'Malegaon', 'Kochi', 'Nagaon', 'Nagpur', 'Srinagar', 'Davanagere',
       'Bhagalpur', 'Siwan', 'Meerut', 'Dindigul', 'Bhatpara',
       'Ghaziabad', 'Kulti', 'Chapra', 'Dibrugarh', 'Panihati',
       'Bhiwandi', 'Morbi', 'Kalyan-Dombivli', 'Gorakhpur', 'Panvel',
       'Siliguri', 'Bongaigaon', 'Patna', 'Ramgarh', 'Ozhukarai',
       'Mirzapur', 'Akola', 'Satna', 'Motihari', 'Jalna', 'Jalandhar',
       'Unnao', 'Karnal', 'Cuttack', 'Proddatur', 'Ichalkaranji',
       'Warangal', 'Jhansi', 'Bulandshahr', 'Narasaraopet', 'Chinsurah',
       'Jehanabad', 'Dhanbad', 'Gudivada', 'Gandhidham', 'Raiganj',
       'Kishanganj', 'Varanasi', 'Belgaum', 'Tirupati', 'Tumkur',
       'Coimbatore', 'Kurnool', 'Gurgaon', 'Muzaffarnagar', 'Bhavnagar',
       'Arrah', 'Munger', 'Tirunelveli', 'Mumbai', 'Mango', 'Nashik',
       'Kadapa', 'Amritsar', 'Khora,_Ghaziabad', 'Ambala', 'Agra',
       'Ratlam', 'Surendranagar_Dudhrej', 'Delhi_city', 'Bhopal', 'Hapur',
       'Rohtak', 'Durg', 'Korba', 'Bangalore', 'Shivpuri', 'Thrissur',
       'Vijayanagaram', 'Farrukhabad', 'Nangloi_Jat', 'Madanapalle',
       'Thoothukudi', 'Nagercoil', 'Gaya', 'Chandigarh_city', 'Jammu',
       'Kakinada', 'Dewas', 'Bhalswa_Jahangir_Pur', 'Baranagar',
       'Firozabad', 'Phusro', 'Allahabad', 'Guna', 'Thane', 'Etawah',
       'Vasai-Virar', 'Pallavaram', 'Morena', 'Ballia', 'Surat',
       'Burhanpur', 'Phagwara', 'Mau', 'Mangalore', 'Alwar',
       'Mahbubnagar', 'Maheshtala', 'Hazaribagh', 'Bihar_Sharif',
       'Faridabad', 'Lucknow', 'Tenali', 'Barasat', 'Amroha', 'Giridih',
       'Begusarai', 'Medininagar', 'Rajahmundry', 'Saharsa', 'New_Delhi',
       'Bhilai', 'Moradabad', 'Machilipatnam', 'Mira-Bhayandar', 'Pali',
       'Navi_Mumbai', 'Mehsana', 'Imphal', 'Kolkata', 'Sambalpur',
       'Ujjain', 'Madhyamgram', 'Jabalpur', 'Jamalpur', 'Ludhiana',
       'Bareilly', 'Gangtok', 'Anand', 'Dehradun', 'Pune', 'Satara',
       'Srikakulam', 'Raipur', 'Jodhpur', 'Darbhanga', 'Nizamabad',
       'Nandyal', 'Dehri', 'Jorhat', 'Ranchi', 'Kumbakonam', 'Guntakal',
       'Haldia', 'Loni', 'Pimpri-Chinchwad', 'Rajkot', 'Nanded', 'Noida',
       'Kirari_Suleman_Nagar', 'Jaunpur', 'Bilaspur', 'Sambhal', 'Dhule',
       'Rourkela', 'Thiruvananthapuram', 'Dharmavaram', 'Nellore',
       'Visakhapatnam', 'Karawal_Nagar', 'Jaipur', 'Avadi', 'Bhimavaram',
       'Bardhaman', 'Silchar', 'Buxar', 'Kavali', 'Tezpur', 'Ramagundam',
       'Yamunanagar', 'Sri_Ganganagar', 'Sasaram', 'Sikar', 'Bally',
       'Bhiwani', 'Rampur', 'Uluberia', 'Sangli-Miraj_&_Kupwad', 'Hosur',
       'Bikaner', 'Shahjahanpur', 'Sultan_Pur_Majra', 'Vijayawada',
       'Bharatpur', 'Tadepalligudem', 'Tinsukia', 'Salem', 'Mathura',
       'Guntur', 'Hubli–Dharwad', 'Guwahati', 'Chittoor', 'Tiruvottiyur',
       'Vadodara', 'Ahmednagar', 'Fatehpur', 'Bhilwara', 'Kharagpur',
       'Bettiah', 'Bhind', 'Bokaro', 'Karaikudi', 'Raebareli',
       'Pudukkottai', 'Udaipur', 'Mysore', 'Panipat', 'Latur',
       'Tadipatri', 'Bahraich', 'Orai', 'Raurkela_Industrial_Township',
       'Gwalior', 'Katni', 'Chandrapur', 'Kolhapur'])
state_encoder.fit(['Madhya_Pradesh', 'Maharashtra', 'Kerala', 'Odisha', 'Tamil_Nadu',
       'Gujarat', 'Rajasthan', 'Telangana', 'Bihar', 'Andhra_Pradesh',
       'West_Bengal', 'Haryana', 'Puducherry', 'Karnataka',
       'Uttar_Pradesh', 'Himachal_Pradesh', 'Punjab', 'Tripura',
       'Uttarakhand', 'Jharkhand', 'Mizoram', 'Assam',
       'Jammu_and_Kashmir', 'Delhi', 'Chhattisgarh', 'Chandigarh',
       'Manipur', 'Sikkim'])

# Define the input data schema
class InputData(BaseModel):
    Income: int
    Age: int
    Experience: int
    Marrital_status: str
    House_Ownership: str
    Car_Ownership: str
    Profession: str
    CITY: str
    STATE: str
    CURRENT_JOB_YRS: int
    CURRENT_HOUSE_YRS: int

# Root endpoint
@app.get('/', response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post('/predict/')
def predict(input_data: InputData):
    # Preprocess categorical variables using LabelEncoder
    marital_status_encoded = marital_status_encoder.transform([input_data.Marrital_status])[0]
    house_ownership_encoded = house_ownership_encoder.transform([input_data.House_Ownership])[0]
    car_ownership_encoded = car_ownership_encoder.transform([input_data.Car_Ownership])[0]
    profession_encoded = profession_encoder.transform([input_data.Profession])[0]
    city_encoded = city_encoder.transform([input_data.CITY])[0]
    state_encoded = state_encoder.transform([input_data.STATE])[0]

    # Combine all input data into a single array for prediction
    data = [
        input_data.Income,
        input_data.Age,
        input_data.Experience,
        marital_status_encoded,
        house_ownership_encoded,
        car_ownership_encoded,
        profession_encoded,
        city_encoded,
        state_encoded,
        input_data.CURRENT_JOB_YRS,
        input_data.CURRENT_HOUSE_YRS
    ]

    # Reshape the data as a 2D array for the model
    data = np.array(data).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(data)

    # Convert prediction to binary output
    prediction_label = "Yes" if prediction[0] == 1 else "No"

    return {"prediction": prediction_label}
