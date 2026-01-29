import unittest

# Desired tests:
# - Only primitives
# - Only simple list
# - Only structured
# - Structured list

from typing import Optional, List, Any

from stickler.structured_object_evaluator.models.structured_model import StructuredModel
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.comparators.numeric import NumericComparator
from stickler.comparators.exact import ExactComparator

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

hungarian_field  = ComparableField(weight=1.0)
  

class LineItemsInfo(StructuredModel):
    LineItemRate: Optional[str] | Any = exact_number
    LineItemDays: Optional[List[str]] | Any = hungarian_field
    LineItemStartDate: Optional[str] | Any = exact_field
    LineItemEndDate: Optional[str] | Any = exact_field
    LineItemDescription: Optional[str] | Any = exact_field

    match_threshold = 1.0

class Invoice(StructuredModel):
    Agency: Optional[str] | Any = exact_field
    Advertiser: Optional[str] | Any = exact_field
    GrossTotal: Optional[str] | Any = exact_number
    PaymentTerms: Optional[str] | Any = exact_field
    AgencyCommission: Optional[str] | Any = exact_number
    NetAmountDue: Optional[str] | Any = exact_number
    LineItems: Optional[List[LineItemsInfo]] | Any = hungarian_field

 
class TestAggregation(unittest.TestCase):
    def test_primitives(self):
        invoice_gt = Invoice(
            Agency= "Great American Media",
            Advertiser= None,
            GrossTotal= "45525.00",
            PaymentTerms= None,
            AgencyCommission= "6828.75",
            NetAmountDue= "38696.25"
        )
        invoice_pred = Invoice(
            Agency= "Great American Media",   #TP
            Advertiser= None,                 #TN
            GrossTotal= "45525.00",           #TP
            PaymentTerms= "Cash",             #FA
            AgencyCommission= None,           #FN
            NetAmountDue= "38696.2"           #FD
                                              #TN: LineItems
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 2, 'tp')
        self.assertEqual(agg_results['tn'], 6, 'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 1, 'fa')
        self.assertEqual(agg_results['fp'], 2, 'fp')
        self.assertEqual(agg_results['fn'], 1, 'fn')

    def test_simple_list(self):
        LineItemsInfo_gt = LineItemsInfo(
            LineItemDays = ['M', 'T', 'W', 'Th', 'F']
        )
        LineItemsInfo_pred = LineItemsInfo(
            LineItemDays = ['M', 'Tuesday', 'Th', 'F'] #TP =3, TN=4, FD=1, FA=0, FP=1, FN=1
        )

        result = LineItemsInfo_gt.compare_with(LineItemsInfo_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 3, 'tp')
        self.assertEqual(agg_results['tn'], 4, 'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 1, 'fp')
        self.assertEqual(agg_results['fn'], 1, 'fn')

    def test_simple_list_empty_gt(self):
        LineItemsInfo_gt = LineItemsInfo(
            LineItemDays = []
        )
        LineItemsInfo_pred = LineItemsInfo(
            LineItemDays = ['M', 'Tuesday', 'Th', 'F'] #TP =3, TN=4, FD=1, FA=0, FP=1, FN=1
        )

        result = LineItemsInfo_gt.compare_with(LineItemsInfo_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 0, 'tp')
        self.assertEqual(agg_results['tn'], 4, 'tn')
        self.assertEqual(agg_results['fd'], 0, 'fd')
        self.assertEqual(agg_results['fa'], 4, 'fa')
        self.assertEqual(agg_results['fp'], 4, 'fp')
        self.assertEqual(agg_results['fn'], 0, 'fn')


    def test_list_structure(self):
        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News",
                LineItemStartDate= "10/11/2016",
                LineItemEndDate= None,
                LineItemRate= "475.00"
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News @ 6a", #FD
                LineItemStartDate= None,                    #FN
                LineItemEndDate= "10/17/2016",              #FA
                LineItemRate= "475.00"                      #TP
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 1, 'tp')
        self.assertEqual(agg_results['tn'], 7, 'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 1, 'fa')
        self.assertEqual(agg_results['fp'], 2, 'fp')
        self.assertEqual(agg_results['fn'], 1, 'fn')


    def test_list_structure_unmatched_gt(self):
        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News",
                LineItemStartDate= "10/11/2016",
                LineItemEndDate= None,
                LineItemRate= "475.00"
            ),
            LineItemsInfo(
                LineItemDescription= "Description2",
                LineItemStartDate= "Date2",
                LineItemEndDate= None,
                LineItemRate= "Rate2"
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News @ 6a", #FD
                LineItemStartDate= None,                    #FN
                LineItemEndDate= "10/17/2016",              #FA
                LineItemRate= "475.00"                      #TP
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 1, 'tp')
        self.assertEqual(agg_results['tn'], 7, 'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 1, 'fa')
        self.assertEqual(agg_results['fp'], 2, 'fp')
        self.assertEqual(agg_results['fn'], 4, 'fn')


    def test_list_structure_unmatched_pred(self):
        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News",
                LineItemStartDate= "10/11/2016",
                LineItemEndDate= None,
                LineItemRate= "475.00"
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News @ 6a", #FD
                LineItemStartDate= None,                    #FN
                LineItemEndDate= "10/17/2016",              #FA
                LineItemRate= "475.00"                      #TP
            ),
            LineItemsInfo(
                LineItemDescription= "Description2",
                LineItemStartDate= "Date2",
                LineItemEndDate= None,
                LineItemRate= "Rate2"
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 1, 'tp')
        self.assertEqual(agg_results['tn'], 7, 'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 4, 'fa')
        self.assertEqual(agg_results['fp'], 5, 'fp')
        self.assertEqual(agg_results['fn'], 1, 'fn')

    def test_list_structure_empty_pred(self):
        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News",
                LineItemStartDate= "10/11/2016",
                LineItemEndDate= None,
                LineItemRate= "475.00"
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = []
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 0, 'tp')
        self.assertEqual(agg_results['tn'], 6, 'tn')
        self.assertEqual(agg_results['fd'], 0, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 0, 'fp')
        self.assertEqual(agg_results['fn'], 3, 'fn')
    
    def test_list_structure_empty_gt(self):
        invoice_gt = Invoice(
            LineItems = []
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDescription= "M-F Local News",
                LineItemStartDate= "10/11/2016",
                LineItemEndDate= None,
                LineItemRate= "475.00"
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 0, 'tp')
        self.assertEqual(agg_results['tn'], 6, 'tn')
        self.assertEqual(agg_results['fd'], 0, 'fd')
        self.assertEqual(agg_results['fa'], 3, 'fa')
        self.assertEqual(agg_results['fp'], 3, 'fp')
        self.assertEqual(agg_results['fn'], 0, 'fn')
    
    def test_simple_list_within_structure(self):

        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'T', 'W', 'Th', 'F'],
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'Tuesday', 'Th', 'F'] #TP =3, TN=4, FD=1, FA=0, FP=1, FN=1
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 3, 'tp')
        self.assertEqual(agg_results['tn'], 10,'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 1, 'fp')
        self.assertEqual(agg_results['fn'], 1, 'fn')
        return
    
    def test_simple_list_within_structure_empty_pred(self):

        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'T', 'W', 'Th', 'F'],
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = []
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 0, 'tp')
        self.assertEqual(agg_results['tn'], 6,'tn')
        self.assertEqual(agg_results['fd'], 0, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 0, 'fp')
        self.assertEqual(agg_results['fn'], 5, 'fn')
        return
    
    def test_simple_list_within_structure_with_duplicates(self):

        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'T', 'W', 'Th', 'F'],
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'Tuesday', 'Th', 'Th', 'F'] #TP = 4(this is also TP: T~=Th due to default threshold of 0.5), FD=1, FP=1
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 4, 'tp')
        self.assertEqual(agg_results['tn'], 10,'tn')
        self.assertEqual(agg_results['fd'], 1, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 1, 'fp')
        self.assertEqual(agg_results['fn'], 0, 'fn')
        return

    def test_simple_list_within_structure_unmatched_gt(self):

        invoice_gt = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['M', 'T', 'W', 'Th', 'F'],
            )
            ]
        )
        invoice_pred = Invoice(
            LineItems = [LineItemsInfo(
                LineItemDays= ['Monday', 'Tuesday', 'Wed', 'Thursday', 'Friday'] #TP=0, TN=0, FD=5, FA=0, FP=0, FN=0
            )
            ]
        )

        result = invoice_gt.compare_with(invoice_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        self.assertEqual(agg_results['tp'], 0, 'tp')
        self.assertEqual(agg_results['tn'], 10,'tn')
        self.assertEqual(agg_results['fd'], 5, 'fd')
        self.assertEqual(agg_results['fa'], 0, 'fa')
        self.assertEqual(agg_results['fp'], 5, 'fp')
        self.assertEqual(agg_results['fn'], 0, 'fn')
        return

if __name__ == '__main__':
    unittest.main()