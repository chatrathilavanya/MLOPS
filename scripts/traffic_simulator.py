import requests
import random
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Traffic simulator for Heart Disease API")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="API URL to send requests to")
    parser.add_argument("--requests", type=int, default=30, help="Number of requests to send")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    args = parser.parse_args()

    # Define a few patient profiles to simulate diverse traffic
    profiles = [
        {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        },
        {
            "age": 37, "sex": 1, "cp": 2, "trestbps": 130,
            "chol": 250, "fbs": 0, "restecg": 1, "thalach": 187,
            "exang": 0, "oldpeak": 3.5, "slope": 0, "ca": 0, "thal": 2
        },
        {
            "age": 41, "sex": 0, "cp": 1, "trestbps": 130,
            "chol": 204, "fbs": 0, "restecg": 0, "thalach": 172,
            "exang": 0, "oldpeak": 1.4, "slope": 2, "ca": 0, "thal": 2
        },
        {
            "age": 56, "sex": 1, "cp": 1, "trestbps": 120,
            "chol": 236, "fbs": 0, "restecg": 1, "thalach": 178,
            "exang": 0, "oldpeak": 0.8, "slope": 2, "ca": 0, "thal": 2
        },
        {
            "age": 57, "sex": 0, "cp": 0, "trestbps": 120,
            "chol": 354, "fbs": 0, "restecg": 1, "thalach": 163,
            "exang": 1, "oldpeak": 0.6, "slope": 2, "ca": 0, "thal": 2
        }
    ]

    print(f"Starting traffic simulation targeting {args.url}")
    print(f"Total requests: {args.requests}, Delay: {args.delay}s")

    success_count = 0
    error_count = 0

    for i in range(args.requests):
        profile = random.choice(profiles)
        try:
            response = requests.post(args.url, json=profile, timeout=5)
            if response.status_code == 200:
                success_count += 1
                print(f"[{i+1}/{args.requests}] SUCCESS - Prediction: {response.json().get('prediction_label')}")
            else:
                error_count += 1
                print(f"[{i+1}/{args.requests}] ERROR {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            error_count += 1
            print(f"[{i+1}/{args.requests}] FAILED - {e}")
        
        time.sleep(args.delay)

    print("\n--- Simulation Complete ---")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {error_count}")

if __name__ == "__main__":
    main()
