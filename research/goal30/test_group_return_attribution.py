"""公司行动中性、成交方向和跨期对数归因的手工算例。"""
import math
import unittest

from research.goal30.group_return_attribution import pnl_components, log_link


class AttributionTests(unittest.TestCase):
    def test_split_price_jump_is_not_loss(self):
        row=pnl_components({'X':100},{'X':10},{'X':5},[{'type':'split','symbol':'X','ratio':2,
            'quantity_before':100,'quantity_after':200}],[])
        self.assertEqual(row['market_price_pnl'],0)
        self.assertEqual(row['split_rounding_pnl'],0)

    def test_dividend_accrues_once_and_payment_is_neutral(self):
        row=pnl_components({'X':100},{'X':10},{'X':9.8},[{'type':'accrue_dividend','amount':20}],[])
        self.assertAlmostEqual(row['market_price_pnl']+row['dividend_accrual_pnl'],0)
        paid=pnl_components({'X':100},{'X':9.8},{'X':9.8},[{'type':'pay_dividend','amount':20}],[])
        self.assertEqual(sum(paid.values()),0)

    def test_integer_rounding_and_buy_sell_timing(self):
        row=pnl_components({'X':101},{'X':10},{'X':20},[{'type':'split','symbol':'X','ratio':.5,
            'quantity_before':101,'quantity_after':51}],[{'side':'sell','symbol':'X','quantity':10,
             'raw_price':19,'commission':5,'slippage_cost':2}])
        self.assertEqual(row['market_price_pnl'],0)
        self.assertEqual(row['split_rounding_pnl'],10)
        self.assertEqual(row['trading_time_pnl'],-10)
        self.assertEqual(row['commission_cost']+row['slippage_cost'],7)

    def test_log_link_compounds_and_handles_equal_returns(self):
        rs=[.1,-.08,.03,0,.02];rb=[.04,-.03,.03,0,-.01]
        linked=sum((s-b)*log_link(s,b) for s,b in zip(rs,rb))
        expected=math.log(math.prod(1+x for x in rs)/math.prod(1+x for x in rb))
        self.assertAlmostEqual(linked,expected,places=13)
        self.assertAlmostEqual(log_link(.1,.1),1/1.1)
        with self.assertRaises(ValueError): log_link(-1,0)


if __name__=='__main__': unittest.main(verbosity=2)
