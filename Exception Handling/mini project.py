# Purchase  (Mini Project)
def process_purchase_file():
    try:
        filename = input("Enter the file name: ").strip()
        with open(filename, "r") as file:
            lines = file.readlines()

        purchased_items = 0
        free_items = 0
        total_amount = 0
        discount = 0

        clean_lines = [line.strip() for line in lines if line.strip()]

        for line in clean_lines[:-1]:  
            try:
                parts = line.split()
                if len(parts) == 2:
                    item, price = parts
                    if price.lower() == "free":
                        free_items += 1
                    else:
                        total_amount += int(price)
                        purchased_items += 1
                else:
                    print(f"Warning: Invalid line format - '{line}'")
            except Exception as e:
                print(f"Error processing line: {line}. Reason: {e}")

        try:
            last_line = clean_lines[-1]
            if last_line.lower().startswith("discount"):
                discount = int(last_line.split()[1])
            else:
                print("No valid discount line found.")
        except:
            print("Error reading discount line.")

        final_amount = total_amount - discount

        print("\n--- Purchase Summary ---")
        print("No of items purchased:", purchased_items)
        print("No of free items:", free_items)
        print("Amount to pay:", total_amount)
        print("Discount given:", discount)
        print("Final amount paid:", final_amount)

    except FileNotFoundError:
        print("Error: File not found. Please check the file name.")
    except Exception as e:
        print("Unexpected error:", e)

process_purchase_file()
