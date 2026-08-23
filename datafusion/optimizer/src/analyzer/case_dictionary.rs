// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`CaseDictionaryEncoding`]: dictionary output for literal-branch CASE

use crate::analyzer::AnalyzerRule;
use crate::analyzer::type_coercion::TypeCoercion;
use crate::utils::NamePreserver;

use arrow::datatypes::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{DFSchema, Result};
use datafusion_expr::expr::{Case, Cast};
use datafusion_expr::utils::merge_schema;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan};

/// Rewrites `CASE` expressions whose THEN and ELSE branches are all literals
/// into `CAST(CASE ... AS Dictionary(UInt32, T))` inside projections. The
/// physical planner recognizes the pattern and emits the branch literals as
/// the dictionary values and the per-row branch choice as the keys, so
/// downstream expressions work per distinct value instead of per row.
///
/// Must run after [`TypeCoercion`], which unifies the branch types; re-runs
/// coercion on its own output so consumers of the now dictionary-typed
/// columns get encoding-aware casts.
///
/// Gated by `datafusion.optimizer.emit_dictionary_for_literal_case`.
#[derive(Default, Debug)]
pub struct CaseDictionaryEncoding {}

impl CaseDictionaryEncoding {
    pub fn new() -> Self {
        Self {}
    }
}

impl AnalyzerRule for CaseDictionaryEncoding {
    fn name(&self) -> &str {
        "case_dictionary_encoding"
    }

    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        if !config.optimizer.emit_dictionary_for_literal_case {
            return Ok(plan);
        }
        let transformed = plan.transform_down_with_subqueries(|plan| match plan {
            // These subtrees answer to schemas fixed independently of the
            // projection expressions: the recursive term must keep matching
            // the work table frozen at SQL-planning time, and DML targets
            // keep their declared column types.
            LogicalPlan::RecursiveQuery(_) | LogicalPlan::Dml(_) => {
                Ok(Transformed::new(plan, false, TreeNodeRecursion::Jump))
            }
            _ => rewrite_projection(plan),
        })?;
        if transformed.transformed {
            TypeCoercion::new().analyze(transformed.data, config)
        } else {
            Ok(transformed.data)
        }
    }
}

fn rewrite_projection(plan: LogicalPlan) -> Result<Transformed<LogicalPlan>> {
    if !matches!(plan, LogicalPlan::Projection(_)) {
        return Ok(Transformed::no(plan));
    }
    let schema = merge_schema(&plan.inputs());
    let name_preserver = NamePreserver::new(&plan);
    let rewritten = plan.map_expressions(|expr| {
        let original_name = name_preserver.save(&expr);
        wrap_literal_cases(expr, &schema)
            .map(|transformed| transformed.update_data(|e| original_name.restore(e)))
    })?;
    if !rewritten.transformed {
        // Leave untouched projections alone: recomputing would discard
        // schemas installed deliberately, such as coerced union metadata.
        return Ok(rewritten);
    }
    rewritten.map_data(|plan| plan.recompute_schema())
}

fn wrap_literal_cases(expr: Expr, schema: &DFSchema) -> Result<Transformed<Expr>> {
    expr.transform_down(|e| {
        // An already wrapped CASE stays as it is.
        if let Expr::Cast(cast) = &e
            && matches!(cast.expr.as_ref(), Expr::Case(_))
            && matches!(cast.field.data_type(), DataType::Dictionary(_, _))
        {
            return Ok(Transformed::new(e, false, TreeNodeRecursion::Jump));
        }
        let Expr::Case(case) = &e else {
            return Ok(Transformed::no(e));
        };
        if !all_branches_are_literals(case) {
            return Ok(Transformed::no(e));
        }
        let value_type = e.get_type(schema)?;
        if !worth_encoding(&value_type) {
            return Ok(Transformed::no(e));
        }
        let dictionary_type =
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(value_type));
        let wrapped = Expr::Cast(Cast::new(Box::new(e), dictionary_type));
        Ok(Transformed::new(wrapped, true, TreeNodeRecursion::Jump))
    })
}

fn all_branches_are_literals(case: &Case) -> bool {
    // Literals carrying field metadata keep the flat path: extension types
    // are out of scope for this rewrite.
    let literal = |e: &Expr| matches!(e, Expr::Literal(_, None));
    !case.when_then_expr.is_empty()
        && case.when_then_expr.iter().all(|(_, then)| literal(then))
        && case.else_expr.as_deref().is_none_or(literal)
}

/// Value types the rewrite admits: dictionary encoding must pay off (Boolean
/// and Null have nothing to gain) and every fallback path must be executable.
/// The physical planner may still plan a plain `CAST` to the dictionary type
/// for shapes the fused expression declines, and arrow cannot pack intervals,
/// durations, or nested types into dictionaries — so those stay flat.
fn worth_encoding(value_type: &DataType) -> bool {
    use DataType::*;
    matches!(
        value_type,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float16
            | Float32
            | Float64
            | Decimal128(_, _)
            | Decimal256(_, _)
            | Date32
            | Date64
            | Time32(_)
            | Time64(_)
            | Timestamp(_, _)
            | Utf8
            | LargeUtf8
            | Utf8View
            | Binary
            | LargeBinary
            | BinaryView
            | FixedSizeBinary(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::test_table_scan;
    use datafusion_expr::{LogicalPlanBuilder, col, lit};

    fn case_label(base: Expr, then: [Expr; 2]) -> Expr {
        let [first, second] = then;
        Expr::Case(Case::new(
            Some(Box::new(base)),
            vec![
                (Box::new(lit(0u32)), Box::new(first)),
                (Box::new(lit(1u32)), Box::new(second)),
            ],
            Some(Box::new(lit("other"))),
        ))
    }

    fn enabled() -> ConfigOptions {
        let mut config = ConfigOptions::default();
        config.optimizer.emit_dictionary_for_literal_case = true;
        config
    }

    fn analyze(plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        CaseDictionaryEncoding::new().analyze(plan, config)
    }

    #[test]
    fn disabled_flag_is_a_noop() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [lit("fizz"), lit("buzz")])])?
            .build()?;
        let before = plan.display_indent().to_string();
        let analyzed = analyze(plan, &ConfigOptions::default())?;
        assert_eq!(before, analyzed.display_indent().to_string());
        Ok(())
    }

    #[test]
    fn wraps_literal_case_in_projection() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [lit("fizz"), lit("buzz")])])?
            .build()?;
        let name_before = plan.schema().field(0).name().clone();
        let analyzed = analyze(plan, &enabled())?;
        let field = analyzed.schema().field(0);
        // The output column keeps its name and reports the dictionary type.
        assert_eq!(field.name(), &name_before);
        assert_eq!(
            field.data_type(),
            &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8))
        );
        let display = analyzed.display_indent().to_string();
        assert!(display.contains("Dictionary(UInt32, Utf8)"), "{display}");
        Ok(())
    }

    #[test]
    fn analyzing_twice_changes_nothing_more() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [lit("fizz"), lit("buzz")])])?
            .build()?;
        let config = enabled();
        let once = analyze(plan, &config)?;
        let display = once.display_indent().to_string();
        let twice = analyze(once, &config)?;
        assert_eq!(display, twice.display_indent().to_string());
        Ok(())
    }

    #[test]
    fn filter_predicates_are_left_alone() -> Result<()> {
        let case = case_label(col("a"), [lit("fizz"), lit("buzz")]);
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .filter(case.eq(lit("fizz")))?
            .project(vec![col("a")])?
            .build()?;
        let before = plan.display_indent().to_string();
        let analyzed = analyze(plan, &enabled())?;
        assert_eq!(before, analyzed.display_indent().to_string());
        Ok(())
    }

    #[test]
    fn non_literal_branches_are_left_alone() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [col("b"), lit(1u32)])])?
            .build()?;
        let before = plan.display_indent().to_string();
        let analyzed = analyze(plan, &enabled())?;
        assert_eq!(before, analyzed.display_indent().to_string());
        Ok(())
    }

    #[test]
    fn interval_output_is_left_alone() -> Result<()> {
        // The fallback path is a plain cast to the dictionary type, and
        // arrow cannot pack intervals into dictionaries.
        let interval =
            |days: i32| lit(datafusion_common::ScalarValue::new_interval_mdn(0, days, 0));
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [interval(1), interval(2)])])?
            .build()?;
        let before = plan.display_indent().to_string();
        let analyzed = analyze(plan, &enabled())?;
        assert_eq!(before, analyzed.display_indent().to_string());
        Ok(())
    }

    #[test]
    fn boolean_output_is_left_alone() -> Result<()> {
        let plan = LogicalPlanBuilder::from(test_table_scan()?)
            .project(vec![case_label(col("a"), [lit(true), lit(false)])])?
            .build()?;
        let before = plan.display_indent().to_string();
        let analyzed = analyze(plan, &enabled())?;
        assert_eq!(before, analyzed.display_indent().to_string());
        Ok(())
    }
}
