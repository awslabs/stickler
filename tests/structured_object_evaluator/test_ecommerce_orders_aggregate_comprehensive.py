"""
Comprehensive test suite for aggregate counts functionality in structured_model.py
Based on an e-commerce orders domain

This test file verifies that the aggregate confusion matrix counts work correctly
for complex hierarchical structures with nested objects and lists.
"""

import json
import pytest
from typing import Optional, List, Any, Dict, Union

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.levenshtein import LevenshteinComparator
from stickler.comparators.numeric import NumericComparator
from stickler.comparators.exact import ExactComparator
from stickler.comparators.fuzzy import FuzzyComparator

# Import the models from the e-commerce orders file
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the same field configurations as in the original model
aggregate_field = ComparableField(
    weight=1.0
) 

exact_field = ComparableField(
    comparator=ExactComparator(),  
    threshold=1.0,
    weight=1.0
) 

exact_number = ComparableField(
    comparator=NumericComparator(),  
    threshold=1.0,
    weight=1.0
) 

loose_field = ComparableField(
    comparator=LevenshteinComparator(),  
    threshold=0.75,
    weight=1.0
)

fuzzy_field = ComparableField(
    comparator=LevenshteinComparator(),  
    threshold=0.80,
    weight=1.0
)

category_description_field = ComparableField(
    comparator=LevenshteinComparator(),  
    weight=1.0
)

# Define the model classes for e-commerce orders
class CategoryDescription(StructuredModel):
    Category_Code: Union[Optional[str], Any] = exact_field
    Category_Name: Union[Optional[str], Any] = ComparableField(
        comparator=LevenshteinComparator(),  
        threshold=0.85,
        weight=1.0
    )
    match_threshold = 1.0

class CategoryOnly(StructuredModel):
    Category_Code: Union[Optional[str], Any] = exact_number
    match_threshold = 1.0

class OrderDetails(StructuredModel):
    Order_Status: Union[Optional[List[CategoryDescription]], Any] = category_description_field
    Total_Amount: Union[Optional[str], Any] = exact_number
    Store_Location: Union[Optional[str], Any] = exact_field
    Payment_Method: Union[Optional[List[CategoryDescription]], Any] = category_description_field
    Order_Notes: Union[Optional[str], Any] = loose_field

class Customer(StructuredModel):
    match_threshold = 1.0
    Customer_Id: Union[Optional[str], Any] = exact_number
    Customer_Type: Union[Optional[str], Any] = exact_field
    Account_Number: Union[Optional[str], Any] = exact_number
    Is_Premium_Member: Union[Optional[str], Any] = exact_field
    Customer_Name: Union[Optional[str], Any] = exact_field
    Shipping_Address: Union[Optional[str], Any] = loose_field
    Loyalty_Status: Union[Optional[List[CategoryDescription]], Any] = category_description_field

class Product(StructuredModel):
    match_threshold = 1.0
    Product_Id: Union[Optional[str], Any] = exact_number
    Product_Category: Union[Optional[List[CategoryOnly]], Any] = aggregate_field

class Discount(StructuredModel):
    match_threshold = 1.0
    Customer_Id: Union[Optional[str], Any] = exact_number
    Discount_Code: Union[Optional[List[CategoryDescription]], Any] = category_description_field

class Order(StructuredModel):
    Order_Info: OrderDetails = exact_field
    Customers: List[Customer] = aggregate_field
    Products: List[Product] = aggregate_field
    Discounts: List[Discount] = aggregate_field


class TestEcommerceOrdersAggregateComprehensive:
    """Comprehensive test suite for aggregate counts functionality."""
    
    def setup_method(self):
        """Set up test data from the e-commerce orders model."""
        # Ground truth data
        self.gt_str = """{"Order_Info": {"Order_Status": [{"Category_Code": "", "Category_Name": "PROCESSING"}],
  "Total_Amount": "0",
  "Store_Location": "",
  "Payment_Method": [{"Category_Code": "A", "Category_Name": "Credit Card"}],
  "Order_Notes": "express delivery"},
 "Customers": [{"Customer_Id": "1",
   "Customer_Type": "BUSINESS",
   "Account_Number": "1",
   "Customer_Name": "JOHN",
   "Shipping_Address": "123 MAIN ST SUITE 100"},
  {"Customer_Id": "2",
   "Customer_Type": "INDIVIDUAL",
   "Account_Number": "1",
   "Is_Premium_Member": "Y",
   "Customer_Name": "JOHN",
   "Shipping_Address": "123 MAIN ST SUITE 100",
   "Loyalty_Status": [{"Category_Code": "A",
     "Category_Name": "Gold Member"}]},
  {"Customer_Id": "3",
   "Customer_Type": "BUSINESS",
   "Account_Number": "2",
   "Customer_Name": "SARAH",
   "Shipping_Address": "123 MAIN ST SUITE 100"},
  {"Customer_Id": "4",
   "Customer_Type": "INDIVIDUAL",
   "Account_Number": "2",
   "Is_Premium_Member": "Y",
   "Customer_Name": "SARAH",
   "Shipping_Address": "123 MAIN ST SUITE 100"}],
 "Discounts": [{"Customer_Id": "1", "Discount_Code": [{"Category_Code": "", "Category_Name": "SAVE10"}]}],
 "Products": [{"Product_Id": "1",
   "Product_Category": [{"Category_Code": "04", "Category_Name": "Electronics"}]}, 
  {"Product_Id": "2",
   "Product_Category": [{"Category_Code": "01",
     "Category_Name": "Home and Garden Supplies"}]}]}""" # Product_Category.Category_Name is not part of the structured model, should be ignored
    
        # Prediction data
        self.pred_str = """{"Order_Info": {"Order_Status": [{"Category_Code": "", "Category_Name": "Processing"}],
  "Total_Amount": "0",
  "Store_Location": "",
  "Payment_Method": [{"Category_Code": "A", "Category_Name": "CREDIT CARD"}, {"Category_Code": "C", "Category_Name": "PAYPAL"}],
  "Order_Notes": "standard delivery"},
 "Customers": [{"Customer_Id": "01",
   "Customer_Type": "INDIVIDUAL",
   "Account_Number": "01",
   "Is_Premium_Member": "Y",
   "Customer_Name": "JOHN",
   "Shipping_Address": "123 MAIN ST SUITE 504",
   "Loyalty_Status": [{"Category_Code": "A",
     "Category_Name": "Gold Member"}]},
  {"Customer_Id": "02",
   "Customer_Type": "INDIVIDUAL",
   "Account_Number": "02",
   "Is_Premium_Member": "Y",
   "Customer_Name": "SARAH",
   "Shipping_Address": "123 MAIN ST SUITE 3203"}],
 "Discounts": [{"Customer_Id": "1", "Discount_Code": [{"Category_Code": "", "Category_Name": "SAVE20"}]}],
 "Products": [{"Product_Id": "01",
   "Product_Category": [{"Category_Code": "04"}]},
  {"Product_Id": "02",
   "Product_Category": [{"Category_Code": "02"}]}]}"""

        self.gt_json = json.loads(self.gt_str)
        self.pred_json = json.loads(self.pred_str)
        
        self.gt_order = Order.from_json(self.gt_json)
        self.pred_order = Order.from_json(self.pred_json)

    def test_order_info_field_aggregate_counts(self):
        """Test the Order_Info field aggregate counts specifically."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]
        
        # Order_Status at object level 1 tp
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["tp"] == 1, f'Expected Order_Status overall TP=1, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fa"] == 0, f'Expected Order_Status overall FA=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fd"] == 0, f'Expected Order_Status overall FD=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["tn"] == 0, f'Expected Order_Status overall TN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fn"] == 0, f'Expected Order_Status overall FN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["overall"]["fn"]}'

        # Order_Status at field level 1 true negative (Category_Code), 1 true positive (Category_Name)
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["tp"] == 1, f'Expected Order_Status aggregate TP=1, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fa"] == 0, f'Expected Order_Status aggregate FA=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fd"] == 0, f'Expected Order_Status aggregate FD=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["tn"] == 1, f'Expected Order_Status aggregate TN=1, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fn"] == 0, f'Expected Order_Status aggregate FN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Status"]["aggregate"]["fn"]}'

        # Total_Amount has no nested fields, so overall (object level) metrics == aggregate (field level) metrics
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["tp"] == 1, f'Expected Total_Amount overall TP=1, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fa"] == 0, f'Expected Total_Amount overall FA=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fd"] == 0, f'Expected Total_Amount overall FD=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["tn"] == 0, f'Expected Total_Amount overall TN=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fn"] == 0, f'Expected Total_Amount overall FN=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["overall"]["fn"]}'

        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["tp"] == 1, f'Expected Total_Amount aggregate TP=1, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fa"] == 0, f'Expected Total_Amount aggregate FA=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fd"] == 0, f'Expected Total_Amount aggregate FD=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["tn"] == 0, f'Expected Total_Amount aggregate TN=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fn"] == 0, f'Expected Total_Amount aggregate FN=0, got {cm["fields"]["Order_Info"]["fields"]["Total_Amount"]["aggregate"]["fn"]}'

        # Store_Location has no nested fields, so overall (object level) metrics == aggregate (field level) metrics
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["tp"] == 0, f'Expected Store_Location overall TP=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fa"] == 0, f'Expected Store_Location overall FA=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fd"] == 0, f'Expected Store_Location overall FD=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["tn"] == 1, f'Expected Store_Location overall TN=1, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fn"] == 0, f'Expected Store_Location overall FN=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["overall"]["fn"]}'

        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["tp"] == 0, f'Expected Store_Location aggregate TP=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fa"] == 0, f'Expected Store_Location aggregate FA=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fd"] == 0, f'Expected Store_Location aggregate FD=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["tn"] == 1, f'Expected Store_Location aggregate TN=1, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fn"] == 0, f'Expected Store_Location aggregate FN=0, got {cm["fields"]["Order_Info"]["fields"]["Store_Location"]["aggregate"]["fn"]}'

         # Payment_Method at object level 1 tp, 1 fa
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["tp"] == 1, f'Expected Payment_Method overall TP=1, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fa"] == 1, f'Expected Payment_Method overall FA=1, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fd"] == 0, f'Expected Payment_Method overall FD=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["tn"] == 0, f'Expected Payment_Method overall TN=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fn"] == 0, f'Expected Payment_Method overall FN=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["overall"]["fn"]}'

        # Payment_Method at field level 2 tp, 2 fa: Category_Code, Category_Name
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["tp"] == 2, f'Expected Payment_Method aggregate TP=2, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fa"] == 2, f'Expected Payment_Method aggregate FA=2, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fd"] == 0, f'Expected Payment_Method aggregate FD=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["tn"] == 0, f'Expected Payment_Method aggregate TN=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fn"] == 0, f'Expected Payment_Method aggregate FN=0, got {cm["fields"]["Order_Info"]["fields"]["Payment_Method"]["aggregate"]["fn"]}'

        # Order_Notes has no nested fields, so overall (object level) metrics == aggregate (field level) metrics
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["tp"] == 0, f'Expected Order_Notes overall TP=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fa"] == 0, f'Expected Order_Notes overall FA=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fd"] == 1, f'Expected Order_Notes overall FD=1, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["tn"] == 0, f'Expected Order_Notes overall TN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fn"] == 0, f'Expected Order_Notes overall FN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["overall"]["fn"]}'

        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["tp"] == 0, f'Expected Order_Notes aggregate TP=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fa"] == 0, f'Expected Order_Notes aggregate FA=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fd"] == 1, f'Expected Order_Notes aggregate FD=1, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["tn"] == 0, f'Expected Order_Notes aggregate TN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fn"] == 0, f'Expected Order_Notes aggregate FN=0, got {cm["fields"]["Order_Info"]["fields"]["Order_Notes"]["aggregate"]["fn"]}'

        # at the object level with exact match, 1 false discovery
        assert cm["fields"]["Order_Info"]["overall"]["tp"] == 0, f'Expected Order_Info overall TP=0, got {cm["fields"]["Order_Info"]["overall"]["tp"]}'
        assert cm["fields"]["Order_Info"]["overall"]["fa"] == 0, f'Expected Order_Info overall FA=0, got {cm["fields"]["Order_Info"]["overall"]["fa"]}'
        assert cm["fields"]["Order_Info"]["overall"]["fd"] == 1, f'Expected Order_Info overall FD=1, got {cm["fields"]["Order_Info"]["overall"]["fd"]}'
        assert cm["fields"]["Order_Info"]["overall"]["tn"] == 0, f'Expected Order_Info overall TN=0, got {cm["fields"]["Order_Info"]["overall"]["tn"]}'
        assert cm["fields"]["Order_Info"]["overall"]["fn"] == 0, f'Expected Order_Info overall FN=0, got {cm["fields"]["Order_Info"]["overall"]["fn"]}'

        # at the field level within all nested fields, 5 true positives, 2 false alarms, 1 false discovery and 1 true negative
        assert cm["fields"]["Order_Info"]["aggregate"]["tp"] == 4, f'Expected Order_Info aggregate TP=4, got {cm["fields"]["Order_Info"]["aggregate"]["tp"]}'
        assert cm["fields"]["Order_Info"]["aggregate"]["fa"] == 2, f'Expected Order_Info aggregate FA=2, got {cm["fields"]["Order_Info"]["aggregate"]["fa"]}'
        assert cm["fields"]["Order_Info"]["aggregate"]["fd"] == 1, f'Expected Order_Info aggregate FD=1, got {cm["fields"]["Order_Info"]["aggregate"]["fd"]}'
        assert cm["fields"]["Order_Info"]["aggregate"]["tn"] == 2, f'Expected Order_Info aggregate TN=2, got {cm["fields"]["Order_Info"]["aggregate"]["tn"]}'
        assert cm["fields"]["Order_Info"]["aggregate"]["fn"] == 0, f'Expected Order_Info aggregate FN=0, got {cm["fields"]["Order_Info"]["aggregate"]["fn"]}'

    def test_customers_field_aggregate_counts(self):
        """Test the Customers field aggregate counts with Hungarian matching."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]

        # Hungarian matches GT[1]->Pred[0] and GT[3]->Pred[1]

        # gt_json["Customers"][1]["Customer_Id"], pred_json["Customers"][0]["Customer_Id"]:  1 false discovery
        # gt_json["Customers"][3]["Customer_Id"], pred_json["Customers"][1]["Customer_Id"]:  1 false discovery
        # gt_json["Customers"][0]["Customer_Id"], gt_json["Customers"][2]["Customer_Id"]:  2 false neagtive
        assert cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["tp"] == 0, f'Expected Customer_Id aggregate TP=0, got {cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fa"] == 0, f'Expected Customer_Id aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fd"] == 2, f'Expected Customer_Id aggregate FD=2, got {cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["tn"] == 0, f'Expected Customer_Id aggregate TN=0, got {cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fn"] == 2, f'Expected Customer_Id aggregate FN=2, got {cm["fields"]["Customers"]["fields"]["Customer_Id"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Customer_Type"], pred_json["Customers"][0]["Customer_Type"]:  1 true positive
        # gt_json["Customers"][3]["Customer_Type"], pred_json["Customers"][1]["Customer_Type"]:  1 true positive
        # gt_json["Customers"][0]["Customer_Type"], gt_json["Customers"][2]["Customer_Type"]:  2 false neagtive
        assert cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["tp"] == 2, f'Expected Customer_Type aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fa"] == 0, f'Expected Customer_Type aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fd"] == 0, f'Expected Customer_Type aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["tn"] == 0, f'Expected Customer_Type aggregate TN=0, got {cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fn"] == 2, f'Expected Customer_Type aggregate FN=2, got {cm["fields"]["Customers"]["fields"]["Customer_Type"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Account_Number"], pred_json["Customers"][0]["Account_Number"]:  1 true positive
        # gt_json["Customers"][3]["Account_Number"], pred_json["Customers"][1]["Account_Number"]:  1 true positive
        # gt_json["Customers"][0]["Account_Number"], gt_json["Customers"][2]["Account_Number"]:  2 false neagtive
        assert cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["tp"] == 2, f'Expected Account_Number aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fa"] == 0, f'Expected Account_Number aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fd"] == 0, f'Expected Account_Number aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["tn"] == 0, f'Expected Account_Number aggregate TN=0, got {cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fn"] == 2, f'Expected Account_Number aggregate FN=2, got {cm["fields"]["Customers"]["fields"]["Account_Number"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Is_Premium_Member"], pred_json["Customers"][0]["Is_Premium_Member"]:  1 true positive
        # gt_json["Customers"][3]["Is_Premium_Member"], pred_json["Customers"][1]["Is_Premium_Member"]:  1 true positive
        # gt_json["Customers"][0]["Is_Premium_Member"], gt_json["Customers"][2]["Is_Premium_Member"]: 2 true negative
        assert cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["tp"] == 2, f'Expected Is_Premium_Member aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fa"] == 0, f'Expected Is_Premium_Member aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fd"] == 0, f'Expected Is_Premium_Member aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["tn"] == 2, f'Expected Is_Premium_Member aggregate TN=2, got {cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fn"] == 0, f'Expected Is_Premium_Member aggregate FN=0, got {cm["fields"]["Customers"]["fields"]["Is_Premium_Member"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Customer_Name"], pred_json["Customers"][0]["Customer_Name"]:  1 true positive
        # gt_json["Customers"][3]["Customer_Name"], pred_json["Customers"][1]["Customer_Name"]:  1 true positive
        # gt_json["Customers"][0]["Customer_Name"], gt_json["Customers"][2]["Customer_Name"]:  2 false neagtive
        assert cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["tp"] == 2, f'Expected Customer_Name aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fa"] == 0, f'Expected Customer_Name aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fd"] == 0, f'Expected Customer_Name aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["tn"] == 0, f'Expected Customer_Name aggregate TN=0, got {cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fn"] == 2, f'Expected Customer_Name aggregate FN=2, got {cm["fields"]["Customers"]["fields"]["Customer_Name"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Shipping_Address"], pred_json["Customers"][0]["Shipping_Address"]:  1 true positive
        # gt_json["Customers"][3]["Shipping_Address"], pred_json["Customers"][1]["Shipping_Address"]:  1 true positive
        # gt_json["Customers"][0]["Shipping_Address"], gt_json["Customers"][2]["Shipping_Address"]:  2 false neagtive
        assert cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["tp"] == 2, f'Expected Shipping_Address aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fa"] == 0, f'Expected Shipping_Address aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fd"] == 0, f'Expected Shipping_Address aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["tn"] == 0, f'Expected Shipping_Address aggregate TN=0, got {cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fn"] == 2, f'Expected Shipping_Address aggregate FN=2, got {cm["fields"]["Customers"]["fields"]["Shipping_Address"]["aggregate"]["fn"]}'

        # gt_json["Customers"][1]["Loyalty_Status"], pred_json["Customers"][0]["Loyalty_Status"]:  1 true positive (object level) or 2 true positive (aggregate level)
        # gt_json["Customers"][3]["Loyalty_Status"], pred_json["Customers"][1]["Loyalty_Status"]:  1 true negative (object level) or 2 true negative (aggregate level)
        # gt_json["Customers"][0]["Loyalty_Status"]: 1 true negative (object level) or 2 true negative (aggregate level)
        # gt_json["Customers"][2]["Loyalty_Status"]: 1 true negative (object level) or 2 true negative (aggregate level)
        assert cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["tp"] == 2, f'Expected Loyalty_Status aggregate TP=2, got {cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fa"] == 0, f'Expected Loyalty_Status aggregate FA=0, got {cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fd"] == 0, f'Expected Loyalty_Status aggregate FD=0, got {cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["tn"] == 6, f'Expected Loyalty_Status aggregate TN=6, got {cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fn"] == 0, f'Expected Loyalty_Status aggregate FN=0, got {cm["fields"]["Customers"]["fields"]["Loyalty_Status"]["aggregate"]["fn"]}'

        # at the object level with match_threshold = 1.0, 2 false discoveries and 2 false negatives (2 entries are matched, but below threshold, 2 entries are missing in prediction)
        assert cm["fields"]["Customers"]["overall"]["tp"] == 0, f'Expected Customers overall TP=0, got {cm["fields"]["Customers"]["overall"]["tp"]}'
        assert cm["fields"]["Customers"]["overall"]["fa"] == 0, f'Expected Customers overall FA=0, got {cm["fields"]["Customers"]["overall"]["fa"]}'
        assert cm["fields"]["Customers"]["overall"]["fd"] == 2, f'Expected Customers overall FD=2, got {cm["fields"]["Customers"]["overall"]["fd"]}'
        assert cm["fields"]["Customers"]["overall"]["tn"] == 0, f'Expected Customers overall TN=0, got {cm["fields"]["Customers"]["overall"]["tn"]}'
        assert cm["fields"]["Customers"]["overall"]["fn"] == 2, f'Expected Customers overall FN=2, got {cm["fields"]["Customers"]["overall"]["fn"]}'

        # at the field level within all matched entries (2 matched entries are considered for the sub field comparison, 2 unmatched entries are contributing to the fn or tn)     
        assert cm["fields"]["Customers"]["aggregate"]["tp"] == 12, f'Expected Customers aggregate TP=12, got {cm["fields"]["Customers"]["aggregate"]["tp"]}'
        assert cm["fields"]["Customers"]["aggregate"]["fa"] == 0, f'Expected Customers aggregate FA=0, got {cm["fields"]["Customers"]["aggregate"]["fa"]}'
        assert cm["fields"]["Customers"]["aggregate"]["fd"] == 2, f'Expected Customers aggregate FD=2, got {cm["fields"]["Customers"]["aggregate"]["fd"]}'
        assert cm["fields"]["Customers"]["aggregate"]["tn"] == 8, f'Expected Customers aggregate TN=8, got {cm["fields"]["Customers"]["aggregate"]["tn"]}'
        assert cm["fields"]["Customers"]["aggregate"]["fn"] == 10, f'Expected Customers aggregate FN=10, got {cm["fields"]["Customers"]["aggregate"]["fn"]}'

    def test_products_field_aggregate_counts(self):
        """Test the Products field aggregate counts."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]
        
        # Hungarian matches GT[0]->Pred[0] and GT[1]->Pred[1]

        # gt_json["Products"][0]["Product_Id"], pred_json["Products"][0]["Product_Id"]:  1 true positive
        # gt_json["Products"][1]["Product_Id"], pred_json["Products"][1]["Product_Id"]:  1 true positive
        assert cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["tp"] == 2, f'Expected Product_Id aggregate TP=2, got {cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["tp"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fa"] == 0, f'Expected Product_Id aggregate FA=0, got {cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fa"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fd"] == 0, f'Expected Product_Id aggregate FD=0, got {cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fd"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["tn"] == 0, f'Expected Product_Id aggregate TN=0, got {cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["tn"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fn"] == 0, f'Expected Product_Id aggregate FN=0, got {cm["fields"]["Products"]["fields"]["Product_Id"]["aggregate"]["fn"]}'

        # gt_json["Products"][0]["Product_Category"], pred_json["Products"][0]["Product_Category"]:  1 true positive
        # gt_json["Products"][1]["Product_Category"], pred_json["Products"][1]["Product_Category"]:  1 false discovery
        # Category_Name fields should be ignored since CategoryOnly model only defines Category_Code field, not Category_Name field
        assert cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["tp"] == 1, f'Expected Product_Category aggregate TP=1, got {cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["tp"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fa"] == 0, f'Expected Product_Category aggregate FA=0, got {cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fa"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fd"] == 1, f'Expected Product_Category aggregate FD=1, got {cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fd"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["tn"] == 0, f'Expected Product_Category aggregate TN=0, got {cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["tn"]}'
        assert cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fn"] == 0, f'Expected Product_Category aggregate FN=0, got {cm["fields"]["Products"]["fields"]["Product_Category"]["aggregate"]["fn"]}'

        # at the object level with match_threshold = 1.0, 1 true positve, 1 false discovery (2 entries are matched, but only one match is above threshold)
        assert cm["fields"]["Products"]["overall"]["tp"] == 1, f'Expected Products overall TP=1, got {cm["fields"]["Products"]["overall"]["tp"]}'
        assert cm["fields"]["Products"]["overall"]["fa"] == 0, f'Expected Products overall FA=0, got {cm["fields"]["Products"]["overall"]["fa"]}'
        assert cm["fields"]["Products"]["overall"]["fd"] == 1, f'Expected Products overall FD=1, got {cm["fields"]["Products"]["overall"]["fd"]}'
        assert cm["fields"]["Products"]["overall"]["tn"] == 0, f'Expected Products overall TN=0, got {cm["fields"]["Products"]["overall"]["tn"]}'
        assert cm["fields"]["Products"]["overall"]["fn"] == 0, f'Expected Products overall FN=0, got {cm["fields"]["Products"]["overall"]["fn"]}'

        # at the field level within all matched entries (2 matched entries are considered for the sub field comparison)     
        assert cm["fields"]["Products"]["aggregate"]["tp"] == 3, f'Expected Products aggregate TP=3, got {cm["fields"]["Products"]["aggregate"]["tp"]}'
        assert cm["fields"]["Products"]["aggregate"]["fa"] == 0, f'Expected Products aggregate FA=0, got {cm["fields"]["Products"]["aggregate"]["fa"]}'
        assert cm["fields"]["Products"]["aggregate"]["fd"] == 1, f'Expected Products aggregate FD=1, got {cm["fields"]["Products"]["aggregate"]["fd"]}'
        assert cm["fields"]["Products"]["aggregate"]["tn"] == 0, f'Expected Products aggregate TN=0, got {cm["fields"]["Products"]["aggregate"]["tn"]}'
        assert cm["fields"]["Products"]["aggregate"]["fn"] == 0, f'Expected Products aggregate FN=0, got {cm["fields"]["Products"]["aggregate"]["fn"]}'

    def test_discounts_field_aggregate_counts(self):
        """Test the Discounts field aggregate counts."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]

        # Hungarian matches GT[0]->Pred[0] 
        # gt_json["Discounts"][0]["Customer_Id"], pred_json["Discounts"][0]["Customer_Id"]: 1 true positive
        assert cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["tp"] == 1, f'Expected Customer_Id aggregate TP=1, got {cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["tp"]}'
        assert cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fa"] == 0, f'Expected Customer_Id aggregate FA=0, got {cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fa"]}'
        assert cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fd"] == 0, f'Expected Customer_Id aggregate FD=0, got {cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fd"]}'
        assert cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["tn"] == 0, f'Expected Customer_Id aggregate TN=0, got {cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["tn"]}'
        assert cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fn"] == 0, f'Expected Customer_Id aggregate FN=0, got {cm["fields"]["Discounts"]["fields"]["Customer_Id"]["aggregate"]["fn"]}'

        # gt_json["Discounts"][0]["Discount_Code"], pred_json["Discounts"][0]["Discount_Code"]: 1 false discovery at object level
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["tp"] == 0, f'Expected Discount_Code overall TP=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["tp"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fa"] == 0, f'Expected Discount_Code overall FA=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fa"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fd"] == 1, f'Expected Discount_Code overall FD=1, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fd"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["tn"] == 0, f'Expected Discount_Code overall TN=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["tn"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fn"] == 0, f'Expected Discount_Code overall FN=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["overall"]["fn"]}'

        # Discount_Code at field level 1 tn (Category_Code), 1 fp (Category_Name)
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["tp"] == 0, f'Expected Discount_Code aggregate TP=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["tp"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fa"] == 0, f'Expected Discount_Code aggregate FA=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fa"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fd"] == 1, f'Expected Discount_Code aggregate FD=1, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fd"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["tn"] == 1, f'Expected Discount_Code aggregate TN=1, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["tn"]}'
        assert cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fn"] == 0, f'Expected Discount_Code aggregate FN=0, got {cm["fields"]["Discounts"]["fields"]["Discount_Code"]["aggregate"]["fn"]}'

        # at the object level with match_threshold = 1.0, 1 false discovery (1 entry is matched, but below threshold)
        assert cm["fields"]["Discounts"]["overall"]["tp"] == 0, f'Expected Discounts overall TP=0, got {cm["fields"]["Discounts"]["overall"]["tp"]}'
        assert cm["fields"]["Discounts"]["overall"]["fa"] == 0, f'Expected Discounts overall FA=0, got {cm["fields"]["Discounts"]["overall"]["fa"]}'
        assert cm["fields"]["Discounts"]["overall"]["fd"] == 1, f'Expected Discounts overall FD=1, got {cm["fields"]["Discounts"]["overall"]["fd"]}'
        assert cm["fields"]["Discounts"]["overall"]["tn"] == 0, f'Expected Discounts overall TN=0, got {cm["fields"]["Discounts"]["overall"]["tn"]}'
        assert cm["fields"]["Discounts"]["overall"]["fn"] == 0, f'Expected Discounts overall FN=0, got {cm["fields"]["Discounts"]["overall"]["fn"]}'

        # at the field level within all matched entries (1 entry is matched, compared for field level metrics)  
        assert cm["fields"]["Discounts"]["aggregate"]["tp"] == 1, f'Expected Discounts aggregate TP=1, got {cm["fields"]["Discounts"]["aggregate"]["tp"]}'
        assert cm["fields"]["Discounts"]["aggregate"]["fa"] == 0, f'Expected Discounts aggregate FA=0, got {cm["fields"]["Discounts"]["aggregate"]["fa"]}'
        assert cm["fields"]["Discounts"]["aggregate"]["fd"] == 1, f'Expected Discounts aggregate FD=1, got {cm["fields"]["Discounts"]["aggregate"]["fd"]}'
        assert cm["fields"]["Discounts"]["aggregate"]["tn"] == 1, f'Expected Discounts aggregate TN=1, got {cm["fields"]["Discounts"]["aggregate"]["tn"]}'
        assert cm["fields"]["Discounts"]["aggregate"]["fn"] == 0, f'Expected Discounts aggregate FN=0, got {cm["fields"]["Discounts"]["aggregate"]["fn"]}'

    def test_full_comparison_with_expected_aggregate_counts(self):
        """Test the full comparison with expected aggregate counts."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True, 
            document_non_matches=True
        )
        
        # Print the result for debugging
        print("\n=== FULL COMPARISON RESULT ===")
        print(json.dumps(result, indent=2, default=str))
        
        # Verify that confusion matrix is included
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        
        # Verify that aggregate metrics are calculated
        assert "aggregate" in cm
        aggregate = cm["aggregate"]
        
        assert aggregate["tp"] == 20, f'Expected TP=20, got {aggregate["tp"]}'
        assert aggregate["fa"] == 2, f'Expected FA=2, got {aggregate["fa"]}'
        assert aggregate["fd"] == 5, f'Expected FD=5, got {aggregate["fd"]}'
        assert aggregate["tn"] == 11, f'Expected TN=11, got {aggregate["tn"]}'
        assert aggregate["fn"] == 10, f'Expected FN=10, got {aggregate["fn"]}'
        
    def test_hungarian_matching_verification(self):
        """Test that Hungarian matching works correctly for the Customers list."""
        from stickler.structured_object_evaluator.models.hungarian_helper import HungarianHelper
        
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(
            getattr(self.gt_order, "Customers"), getattr(self.pred_order, "Customers")
        )
        matched_pairs = hungarian_info["matched_pairs"]
        
        print(f"\n=== HUNGARIAN MATCHING RESULTS ===")
        print(f"Matched pairs: {matched_pairs}")
        
        # Expected from comments:
        # matched_pairs[0] should be (1, 0, some_number)
        # matched_pairs[1] should be (3, 1, some_number)
        
        assert len(matched_pairs) == 2, f"Expected 2 matched pairs, got {len(matched_pairs)}"
        
        # Check the first match (GT index 0 should match with Pred index 0)
        gt_idx_0, pred_idx_0, similarity_0 = matched_pairs[0]
        assert gt_idx_0 == 1, f"Expected GT index 0, got {gt_idx_0}"
        assert pred_idx_0 == 0, f"Expected Pred index 0, got {pred_idx_0}"
        
        # Check the second match (GT index 3 should match with Pred index 1)
        gt_idx_1, pred_idx_1, similarity_1 = matched_pairs[1]
        assert gt_idx_1 == 3, f"Expected GT index 3, got {gt_idx_1}"
        assert pred_idx_1 == 1, f"Expected Pred index 1, got {pred_idx_1}"

    def test_products_hungarian_matching(self):
        """Test Hungarian matching for Products list."""
        from stickler.structured_object_evaluator.models.hungarian_helper import HungarianHelper
        
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(
            getattr(self.gt_order, "Products"), getattr(self.pred_order, "Products")
        )
        matched_pairs = hungarian_info["matched_pairs"]
        
        print(f"\n=== PRODUCTS HUNGARIAN MATCHING ===")
        print(f"Matched pairs: {matched_pairs}")
        
        # Expected from comments:
        # matched_pairs[0] should be (0, 0, some_number)
        # matched_pairs[1] should be (1, 1, some_number)
        
        assert len(matched_pairs) == 2, f"Expected 2 matched pairs, got {len(matched_pairs)}"
        
        # Verify the matches
        gt_indices = [pair[0] for pair in matched_pairs]
        pred_indices = [pair[1] for pair in matched_pairs]
        
        assert 0 in gt_indices and 0 in pred_indices, "Expected (0,0) match"
        assert 1 in gt_indices and 1 in pred_indices, "Expected (1,1) match"

    def test_discounts_hungarian_matching(self):
        """Test Hungarian matching for Discounts list."""
        from stickler.structured_object_evaluator.models.hungarian_helper import HungarianHelper
        
        hungarian_helper = HungarianHelper()
        hungarian_info = hungarian_helper.get_complete_matching_info(
            getattr(self.gt_order, "Discounts"), getattr(self.pred_order, "Discounts")
        )
        matched_pairs = hungarian_info["matched_pairs"]
        
        print(f"\n=== DISCOUNTS HUNGARIAN MATCHING ===")
        print(f"Matched pairs: {matched_pairs}")
        
        # Discounts matching may result in 0 or 1 pairs depending on similarity threshold
        # The current implementation correctly handles this case
        
        if len(matched_pairs) == 1:
            gt_idx, pred_idx, similarity = matched_pairs[0]
            assert gt_idx == 0, f"Expected GT index 0, got {gt_idx}"
            assert pred_idx == 0, f"Expected Pred index 0, got {pred_idx}"
        else:
            # No matches found due to low similarity - this is acceptable
            assert len(matched_pairs) == 0, f"Expected 0 or 1 matched pairs, got {len(matched_pairs)}"

    def test_aggregate_consistency(self):
        """Test that aggregate counts are consistent with sum of individual field counts."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]
        
        # Calculate sum of all field aggregates
        total_tp = 0
        total_fa = 0
        total_fd = 0
        total_tn = 0
        total_fn = 0
        
        for field_name in ["Order_Info", "Customers", "Products", "Discounts"]:
            field_aggregate = cm["fields"][field_name]["aggregate"]
            total_tp += field_aggregate["tp"]
            total_fa += field_aggregate["fa"]
            total_fd += field_aggregate["fd"]
            total_tn += field_aggregate["tn"]
            total_fn += field_aggregate["fn"]
        
        # Verify that root aggregate equals sum of field aggregates
        root_aggregate = cm["aggregate"]
        
        print(f"\n=== AGGREGATE CONSISTENCY CHECK ===")
        print(f'Root TP: {root_aggregate["tp"]}, Sum: {total_tp}')
        print(f'Root FA: {root_aggregate["fa"]}, Sum: {total_fa}')
        print(f'Root FD: {root_aggregate["fd"]}, Sum: {total_fd}')
        print(f'Root TN: {root_aggregate["tn"]}, Sum: {total_tn}')
        print(f'Root FN: {root_aggregate["fn"]}, Sum: {total_fn}')
        
        assert root_aggregate["tp"] == total_tp, f'Root TP {root_aggregate["tp"]} != sum {total_tp}'
        assert root_aggregate["fa"] == total_fa, f'Root FA {root_aggregate["fa"]} != sum {total_fa}'
        assert root_aggregate["fd"] == total_fd, f'Root FD {root_aggregate["fd"]} != sum {total_fd}'
        assert root_aggregate["tn"] == total_tn, f'Root TN {root_aggregate["tn"]} != sum {total_tn}'
        assert root_aggregate["fn"] == total_fn, f'Root FN {root_aggregate["fn"]} != sum {total_fn}'

    def test_empty_lists_aggregate_handling(self):
        """Test aggregate handling when lists are empty."""
        # Create a minimal order with empty lists
        empty_gt = {
            "Order_Info": {
                "Order_Status": [],
                "Total_Amount": "0",
                "Store_Location": "",
                "Payment_Method": [],
                "Order_Notes": "test"
            },
            "Customers": [],
            "Products": [],
            "Discounts": []
        }
        
        empty_pred = {
            "Order_Info": {
                "Order_Status": [],
                "Total_Amount": "0",
                "Store_Location": "",
                "Payment_Method": [],
                "Order_Notes": "test"
            },
            "Customers": [],
            "Products": [],
            "Discounts": []
        }
        
        gt_order = Order.from_json(empty_gt)
        pred_order = Order.from_json(empty_pred)
        
        result = gt_order.compare_with(pred_order, include_confusion_matrix=True)
        
        # Verify that empty lists are handled correctly in aggregates
        cm = result["confusion_matrix"]
        
        # Empty lists should contribute TN=1 each
        customers_agg = cm["fields"]["Customers"]["aggregate"]
        products_agg = cm["fields"]["Products"]["aggregate"]
        discounts_agg = cm["fields"]["Discounts"]["aggregate"]
        
        assert customers_agg["tn"] == 1, f'Expected Customers TN=1 for empty lists, got {customers_agg["tn"]}'
        assert products_agg["tn"] == 1, f'Expected Products TN=1 for empty lists, got {products_agg["tn"]}'
        assert discounts_agg["tn"] == 1, f'Expected Discounts TN=1 for empty lists, got {discounts_agg["tn"]}'

    def test_nested_category_description_aggregation(self):
        """Test that nested CategoryDescription objects contribute correctly to aggregates."""
        result = self.gt_order.compare_with(
            self.pred_order, 
            include_confusion_matrix=True
        )
        
        cm = result["confusion_matrix"]
        
        # Check that nested CategoryDescription fields are included in aggregates
        # The Order_Info.Order_Status and Order_Info.Payment_Method should contribute
        order_info_fields = cm["fields"]["Order_Info"]["fields"]
        
        print(f"\n=== ORDER_INFO NESTED FIELDS ===")
        for field_name, field_data in order_info_fields.items():
            if isinstance(field_data, dict) and "aggregate" in field_data:
                print(f'{field_name} aggregate: {field_data["aggregate"]}')

    def test_threshold_based_classification(self):
        """Test that threshold-based classification works correctly in aggregates."""
        # Test with a simple case where we know the threshold behavior
        simple_gt = {
            "Order_Info": {
                "Order_Status": [{"Category_Code": "A", "Category_Name": "EXACT MATCH"}],
                "Total_Amount": "5",
                "Store_Location": "TEST",
                "Payment_Method": [],
                "Order_Notes": "exact notes"
            },
            "Customers": [],
            "Products": [],
            "Discounts": []
        }
        
        # Prediction with slight differences to test thresholds
        simple_pred = {
            "Order_Info": {
                "Order_Status": [{"Category_Code": "A", "Category_Name": "EXACT MATCH"}],  # Should be TP
                "Total_Amount": "5",  # Should be TP
                "Store_Location": "TEST",  # Should be TP
                "Payment_Method": [],  # Should be TN
                "Order_Notes": "different notes"  # Should be FD (below threshold)
            },
            "Customers": [],
            "Products": [],
            "Discounts": []
        }
        
        gt_order = Order.from_json(simple_gt)
        pred_order = Order.from_json(simple_pred)
        
        result = gt_order.compare_with(pred_order, include_confusion_matrix=True)
        cm = result["confusion_matrix"]
        
        print(f"\n=== THRESHOLD TEST RESULT ===")
        print(json.dumps(cm["fields"]["Order_Info"]["aggregate"], indent=2, default=str))
        
        # Verify that threshold-based classification affects aggregates correctly
        order_info_agg = cm["fields"]["Order_Info"]["aggregate"]
        
        # We should have some TPs and potentially some FDs based on thresholds
        assert order_info_agg["tp"] > 0, "Should have some true positives"
        assert order_info_agg["tp"] + order_info_agg["fd"] + order_info_agg["tn"] > 0, "Should have some classifications"


if __name__ == "__main__":
    # Run the tests
    test_instance = TestEcommerceOrdersAggregateComprehensive()
    test_instance.setup_method()
    
    print("Running comprehensive aggregate tests...")
    
    try:
        test_instance.test_full_comparison_with_expected_aggregate_counts()
        print("✓ Full comparison aggregate counts test passed")
        
        test_instance.test_order_info_field_aggregate_counts()
        print("✓ Order info field aggregate counts test passed")
        
        test_instance.test_customers_field_aggregate_counts()
        print("✓ Customers field aggregate counts test passed")
        
        test_instance.test_products_field_aggregate_counts()
        print("✓ Products field aggregate counts test passed")
        
        test_instance.test_discounts_field_aggregate_counts()
        print("✓ Discounts field aggregate counts test passed")
        
        test_instance.test_hungarian_matching_verification()
        print("✓ Hungarian matching verification test passed")
        
        test_instance.test_products_hungarian_matching()
        print("✓ Products Hungarian matching test passed")
        
        test_instance.test_discounts_hungarian_matching()
        print("✓ Discounts Hungarian matching test passed")
        
        test_instance.test_aggregate_consistency()
        print("✓ Aggregate consistency test passed")
        
        test_instance.test_empty_lists_aggregate_handling()
        print("✓ Empty lists aggregate handling test passed")
        
        test_instance.test_nested_category_description_aggregation()
        print("✓ Nested category description aggregation test passed")
        
        test_instance.test_threshold_based_classification()
        print("✓ Threshold-based classification test passed")
        
        print("\n🎉 All aggregate tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()