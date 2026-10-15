"""独立行为夹具，不调整已登记策略参数。"""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.two_line import dynamic_industry as d


class SelectionTests(unittest.TestCase):
    def test_boundary_tie_must_not_buy_lower_rank(self):
        symbols=['SH512010','SH512400','SH515880','SH512800','SZ159928',d.GOLD]
        dates=pd.bdate_range('2020-01-01',periods=220)
        prices=pd.DataFrame({s:100*np.exp(np.arange(220)*.001) for s in symbols},index=dates)
        scores=pd.DataFrame([[.4,.3,.2,.2,.1,0]],index=[dates[-1]],columns=symbols)
        trend=pd.DataFrame(True,index=[dates[-1]],columns=symbols)
        with patch.object(d,'features',return_value=(scores,trend)):
            result,_=d.targets_for('dynamic',prices,set(symbols),
                {'SH512010':.2,'SH512400':.2},{'SH512010':1,'SH512400':1},[])
        self.assertEqual({s for s in result if d.equity(s)}, {'SH512010','SH512400'})
        self.assertNotIn('SZ159928',result)

    def test_cash_and_prefix(self):
        self.assertTrue(d.checks()['passed'])


if __name__=='__main__':
    unittest.main()
