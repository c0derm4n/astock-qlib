"""周频迁移实验以正的损失幅度检验最大回撤，并联合检查用户目标。"""
import unittest

from research.goal30.weekly_transfer import goal_checks


class WeeklyTransferGoalChecksTest(unittest.TestCase):
    GOALS = {
        "conservative": {"cagr": 0.05, "worst_complete_year": -0.05},
        "advanced": {"cagr": 0.135, "max_drawdown": 0.40},
    }

    def test_rejects_loss_larger_than_drawdown_limit(self):
        metrics = {"cagr": 0.16, "max_drawdown": -0.45,
                   "worst_complete_year": {"year": 2022, "return": -0.03}}
        result = goal_checks(metrics, self.GOALS)
        self.assertFalse(result["advanced_met"])
        self.assertFalse(result["joint_met"])

    def test_annual_limit_remains_required_for_joint_goal(self):
        metrics = {"cagr": 0.16, "max_drawdown": -0.25,
                   "worst_complete_year": {"year": 2022, "return": -0.13}}
        result = goal_checks(metrics, self.GOALS)
        self.assertTrue(result["advanced_met"])
        self.assertFalse(result["joint_met"])
        metrics["worst_complete_year"]["return"] = -0.04
        self.assertTrue(goal_checks(metrics, self.GOALS)["joint_met"])


if __name__ == "__main__":
    unittest.main()
