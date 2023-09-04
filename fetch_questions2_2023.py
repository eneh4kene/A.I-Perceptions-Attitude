import requests
import json
import time
import datetime


def fetch_stack_overflow_data(all_keywords, api_key=None, file_name=None, end_date=None):
    base_url = "https://api.stackexchange.com/2.3/questions?"
    end_date = datetime.datetime(2023, 8, 25, 0, 16)  # 12:16am, 25th August, 2023
    start_date = end_date - datetime.timedelta(days=365)  # Last 365 days

    questions_list = []
    base_params = {'key': api_key} if api_key else {}

    for keyword in all_keywords:
        print(f"Fetching data for keyword: {keyword}")
        page = 1
        has_more = True

        while has_more:
            params = {
                "order": "desc",
                "sort": "activity",
                "tagged": keyword,
                "site": "stackoverflow",
                "fromdate": int(start_date.timestamp()),  # Convert to Unix timestamp
                "todate": int(end_date.timestamp()),  # Convert to Unix timestamp
                "page": page,
                **base_params
            }

            response = requests.get(base_url, params=params)

            if response.status_code == 200:
                questions_data = json.loads(response.text)
                questions_list.extend(questions_data["items"])

                print(f"Fetched page {page} for keyword {keyword}")  # <-- Print statement here

                # Check if more data is available
                has_more = questions_data.get('has_more', False)

                # Update rate-limiting and quota from headers
                remaining_quota = int(response.headers.get('X-RateLimit-Remaining', 0))

                if remaining_quota == 0:
                    print("Rate limit reached. Waiting...")
                    time.sleep(1)  # Wait for a minute or a suitable duration

                # Only increment the page number if there are more items to fetch
                if has_more:
                    page += 1

                # Sleep to respect rate limits
                time.sleep(1)  # 1-second delay to prevent hitting the rate limit

            else:
                print(f"Failed to fetch data for keyword: {keyword}")
                break


    with open(file_name, "w") as f:
        json.dump(questions_list, f)

    print("Data collection completed and saved.")


if __name__ == "__main__":
    api_key = "0sB4SwvMfiDvoJhWXiONGg(("
    tags = ["artificial-intelligence"]
    file_name_2023 = "stackoverflow_question_2023.json"
    file_name_2022 = "stackoverflow_question_2022.json"
    fetch_stack_overflow_data(tags, api_key, file_name_2023)
    fetch_stack_overflow_data(tags, api_key, file_name_2022)
