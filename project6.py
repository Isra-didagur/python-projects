from forex_python.converter import CurrencyRates, CurrencyConverter
import requests

def convert_currency():
    try:
        # Try using CurrencyConverter first (uses cached rates)
        c = CurrencyConverter()
        
        # Get user inputs
        try:
            amount = float(input("Enter the amount: "))
            if amount <= 0:
                print("Please enter a positive amount.")
                return
        except ValueError:
            print("Invalid amount. Please enter a numeric value.")
            return
            
        from_currency = input("Enter the from currency (eg: USD): ").upper()
        to_currency = input("Enter the to currency (eg: INR): ").upper()
        
        # Attempt conversion
        try:
            convert = c.convert(from_currency, to_currency, amount)
            print(f"{amount} {from_currency} is equal to {convert:.2f} {to_currency}")
            
            # Ask about reverse conversion
            print("Do you want to reverse the conversion?")
            reverse = input("Yes or No: ").lower()
            if reverse == "yes":
                rev_convert = c.convert(to_currency, from_currency, convert)
                print(f"{convert:.2f} {to_currency} is equal to {rev_convert:.2f} {from_currency}")
            else:
                print("Thank you for using our currency converter!")
                
        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Please check that you've entered valid currency codes.")
            print("Examples: USD, EUR, GBP, INR, JPY, CAD, AUD, etc.")
            
    except requests.exceptions.RequestException:
        print("Network error: Unable to connect to the currency service.")
        print("Please check your internet connection and try again later.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        
if __name__ == "__main__":
    convert_currency()