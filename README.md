# mlmodel
Loan Prediction based on Customer behaviour

- Open terminal and navigate to the project directory, then run the following command to build the Docker image
  docker build -t image .
  
- After the image is built, run the following command to start the Docker container
  docker run --name container_name -p 8000:8000 image

- Once the container is running, you can access the application in your web browser at
  http://127.0.0.1:8000/
  
