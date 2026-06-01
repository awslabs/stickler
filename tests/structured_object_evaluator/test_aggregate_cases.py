# Desired tests:
# - Only primitives
# - Only simple list
# - Only structured
# - Structured list

from typing import Any, List, Optional

from stickler.comparators.exact import ExactComparator
from stickler.comparators.numeric import NumericComparator
from stickler.structured_object_evaluator.models.comparable_field import ComparableField
from stickler.structured_object_evaluator.models.structured_model import StructuredModel

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

 
class TestAggregation:
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

        assert agg_results['tp'] == 2, 'tp'
        assert agg_results['tn'] == 6, 'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 1, 'fa'
        assert agg_results['fp'] == 2, 'fp'
        assert agg_results['fn'] == 1, 'fn'

    def test_simple_list(self):
        LineItemsInfo_gt = LineItemsInfo(
            LineItemDays = ['M', 'T', 'W', 'Th', 'F']
        )
        LineItemsInfo_pred = LineItemsInfo(
            LineItemDays = ['M', 'Tuesday', 'Th', 'F'] #TP =3, TN=4, FD=1, FA=0, FP=1, FN=1
        )

        result = LineItemsInfo_gt.compare_with(LineItemsInfo_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        assert agg_results['tp'] == 3, 'tp'
        assert agg_results['tn'] == 4, 'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 1, 'fp'
        assert agg_results['fn'] == 1, 'fn'

    def test_simple_list_empty_gt(self):
        LineItemsInfo_gt = LineItemsInfo(
            LineItemDays = []
        )
        LineItemsInfo_pred = LineItemsInfo(
            LineItemDays = ['M', 'Tuesday', 'Th', 'F'] #TP =3, TN=4, FD=1, FA=0, FP=1, FN=1
        )

        result = LineItemsInfo_gt.compare_with(LineItemsInfo_pred, include_confusion_matrix=True)
        agg_results = result['confusion_matrix']['aggregate']

        assert agg_results['tp'] == 0, 'tp'
        assert agg_results['tn'] == 4, 'tn'
        assert agg_results['fd'] == 0, 'fd'
        assert agg_results['fa'] == 4, 'fa'
        assert agg_results['fp'] == 4, 'fp'
        assert agg_results['fn'] == 0, 'fn'

    def test_primitive_list_zero_similarity_treated_as_unmatched(self):
        """Regression test: single-item primitive lists with zero similarity
        must be treated as both items unmatched (FN + FA), not as a paired-but-
        mismatched item (FD).

        Previously the HungarianMatcher single-item shortcut returned a
        synthetic matched pair for score==0, which caused the
        ``unordered_list_metrics`` helper to count the pair as a False
        Discovery (fd:1, fa:0, fn:0). The correct behavior — consistent with
        StructuredListComparator's zero-similarity handling — is to leave the
        items unmatched, yielding fa:1, fn:1, fd:0.

        See PR #115 review feedback.
        """
        from typing import List, Any, Optional

        class Doc(StructuredModel):
            tags: Optional[List[str]] | Any = exact_field

        gt = Doc(tags=['a'])
        pred = Doc(tags=['b'])

        result = gt.compare_with(pred, include_confusion_matrix=True)
        tags_overall = result['confusion_matrix']['fields']['tags']['overall']

        assert tags_overall['fa'] == 1, f"fa: expected 1, got {tags_overall['fa']}"
        assert tags_overall['fn'] == 1, f"fn: expected 1, got {tags_overall['fn']}"
        assert tags_overall['fd'] == 0, f"fd: expected 0 (regression — used to be 1), got {tags_overall['fd']}"
        assert tags_overall['tp'] == 0, f"tp: expected 0, got {tags_overall['tp']}"
        # fp = fa + fd should remain consistent
        assert tags_overall['fp'] == tags_overall['fa'] + tags_overall['fd'], 'fp = fa + fd'


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

        assert agg_results['tp'] == 1, 'tp'
        assert agg_results['tn'] == 7, 'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 1, 'fa'
        assert agg_results['fp'] == 2, 'fp'
        assert agg_results['fn'] == 1, 'fn'


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

        assert agg_results['tp'] == 1, 'tp'
        assert agg_results['tn'] == 7, 'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 1, 'fa'
        assert agg_results['fp'] == 2, 'fp'
        assert agg_results['fn'] == 4, 'fn'


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

        assert agg_results['tp'] == 1, 'tp'
        assert agg_results['tn'] == 7, 'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 4, 'fa'
        assert agg_results['fp'] == 5, 'fp'
        assert agg_results['fn'] == 1, 'fn'

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

        assert agg_results['tp'] == 0, 'tp'
        assert agg_results['tn'] == 6, 'tn'
        assert agg_results['fd'] == 0, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 0, 'fp'
        assert agg_results['fn'] == 3, 'fn'
    
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

        assert agg_results['tp'] == 0, 'tp'
        assert agg_results['tn'] == 6, 'tn'
        assert agg_results['fd'] == 0, 'fd'
        assert agg_results['fa'] == 3, 'fa'
        assert agg_results['fp'] == 3, 'fp'
        assert agg_results['fn'] == 0, 'fn'
    
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

        assert agg_results['tp'] == 3, 'tp'
        assert agg_results['tn'] == 10,'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 1, 'fp'
        assert agg_results['fn'] == 1, 'fn'
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

        assert agg_results['tp'] == 0, 'tp'
        assert agg_results['tn'] == 6,'tn'
        assert agg_results['fd'] == 0, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 0, 'fp'
        assert agg_results['fn'] == 5, 'fn'
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

        assert agg_results['tp'] == 4, 'tp'
        assert agg_results['tn'] == 10,'tn'
        assert agg_results['fd'] == 1, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 1, 'fp'
        assert agg_results['fn'] == 0, 'fn'
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

        assert agg_results['tp'] == 0, 'tp'
        assert agg_results['tn'] == 10,'tn'
        assert agg_results['fd'] == 5, 'fd'
        assert agg_results['fa'] == 0, 'fa'
        assert agg_results['fp'] == 5, 'fp'
        assert agg_results['fn'] == 0, 'fn'
        return

if __name__ == '__main__':
    # Run the tests
    test_instance = TestAggregation()

    print("Running aggregate tests...")
    
    try:
        test_instance.test_primitives()
        test_instance.test_simple_list()
        test_instance.test_simple_list_empty_gt()
        test_instance.test_list_structure()
        test_instance.test_list_structure_unmatched_gt()
        test_instance.test_list_structure_unmatched_pred()
        test_instance.test_list_structure_empty_pred()
        test_instance.test_list_structure_empty_gt()
        test_instance.test_simple_list_within_structure()
        test_instance.test_simple_list_within_structure_empty_pred()
        test_instance.test_simple_list_within_structure_with_duplicates()
        test_instance.test_simple_list_within_structure_unmatched_gt()
        
        
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()