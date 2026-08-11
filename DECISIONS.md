# Decisions and corrections

Two records make up this document, the parameters fixed before the backtest and the corrections
applied during it, with the complete account held entry by entry in Section 13 of `SPEC.md`. Fixing
every parameter in advance of any result rules out specification search as the source of the
near-null finding the study reports.

## Frozen parameters

I fixed the parameters below in `config/config.yaml` and `SPEC.md` before running the first backtest
and left them unchanged once estimation began.

| parameter | value |
|---|---|
| Underlying | QQQ, total return (dividends reinvested) |
| Evaluation period | 2000-01-03 to the most recent complete month |
| Rebalance frequency | monthly, at the close of the last trading day |
| Target volatility | 20% annualized |
| Leverage cap / floor | 2.0× / 0.0× (no shorting) |
| Starting capital | $100,000 |
| Borrow rate | effective fed funds (FRED `DFF`) + 50 bp |
| Cash credit rate | effective fed funds |
| Transaction cost | 2 bp round trip on traded notional |
| Forecast horizon | h = 21 trading days |
| Realized-variance proxy | daily Yang-Zhang (5-minute intraday deferred; see the note below) |
| Seven models | GARCH(1,1), EGARCH, GJR-GARCH, EWMA, trailing realized variance, HAR-RV, rough volatility |

The sensitivity grid varies the volatility target over {10, 15, 20, 25}%, the leverage cap over
{1.5, 2.0, 3.0}, and the borrow spread over {0, 50, 150} bp. I run the grid only after the main study
and report every cell, so the alternatives serve as robustness checks and never feed back into the
headline result.

**The realized-variance fallback.** A single element of the design changed during the work, the
realized-variance measure. Isolating one ticker's 5-minute bars from the candidate source would have
required downloading its full 84 GB archive, so the study adopts the daily Yang-Zhang estimator named
as the fallback in `SPEC.md` Section 3.4, and the frozen parameters above remain in force.

## Corrections

Each entry below records a claim I advanced, later found wrong, and revised, and Section 13 of
`SPEC.md` retains the superseded version of each so that no record of an error has been removed.

**The rough-volatility retransformation constant.** The rough-volatility model retransforms a
forecast of log variance into a variance, and I initially fixed the conditional-variance constant of
that step at 0.5. The value implied by the process is instead the fractional-Brownian constant c(H),
and 0.5 is only its H → 0 limit. Adopting c(H) moved the mean ratio of forecast to realized variance
from 0.81 to 0.86 and the Mincer-Zarnowitz slope from 1.43 to 1.35, though I made the change on
theoretical grounds alone, c(H) being the constant the fractional process implies at any Hurst value.
A residual variance-level bias of 12% to 14% remains, and I report it as a property of the
forecaster.

**The adherence headline.** The largest correction concerns the adherence headline. I first read the
economic-layer separation as a division between resizing and not resizing. A stationary bootstrap
then indicated a second, finer division in which the dynamic models outperformed the naive resizers,
but that reading grouped rough volatility with the naive strategies on the basis of its adherence
statistic alone, a circular step given that rough volatility possesses conditional dynamics and
enters the forecast-accuracy confidence set. Regrouping the models by structure reduced the finer
division to an effect confined to the shorter bootstrap block lengths, covering four models and
excluding rough volatility, and returned the headline close to its original form.

**The rough-volatility mechanism.** I advanced a mechanism and briefly reported it as a result,
namely that the variance-level bias of rough volatility drives an otherwise credible forecaster to
adhere poorly. The claim did not survive a closer check. The two supporting calculations had measured
different bias objects, one of them placing realized volatility on the wrong side of the target, and
the like-for-like signed deviation runs directly against the mechanism, so I returned the claim to a
conjecture and removed the numerical coincidence on which it had rested. Rough volatility forecasts
credibly yet falls outside the adherence set, and the mechanism itself remains a possibility I have
not established.

**The signed target deviation.** An earlier account attributed the block-level volatility undershoot
entirely to the concavity of the square root, an explanation that fails for the rough-volatility and
HAR pair, since rough volatility undershoots the target by less than HAR despite the larger concavity
gap. Decomposing the signed deviation into a forecast-level offset and a concavity term recovers the
correct ordering once both are included, and the Layer-2 section now presents the full decomposition
without disturbing the separate dispersion finding.

**The Harvey (2018) comparison.** An earlier volatility-of-volatility comparison placed my raw figures
beside Harvey (2018) without adjusting for his lower volatility target. Dividing each figure by its
target yields the coefficient of variation, the quantity that makes the two studies comparable, and
on that basis the reduction here is 2.25× against Harvey's 2.6×. The two are close, though the
uncorrected levels had suggested a near-halving.

**HAR's standing.** Excluding the 2008-2009 and 2020 crisis windows moved HAR's forecast-accuracy
rank by several places while leaving its membership in the leading group intact. I therefore treat
the four accurate models as an unordered set, since their internal order depends on the sample, and
only the separation between that set and the naive measures holds across the robustness cuts.

**The bias label.** Earlier drafts characterized the residual rough-volatility bias as though it
distorted position size, an inversion of the actual arithmetic. The 12% to 14% figure applies at the
variance level, whereas the sizing rule operates on volatility, where the aggregate forecast-to-realized
ratio of 1.021 leaves the forecast very nearly unbiased. I relabeled the bias as variance-level
throughout and deleted the single sentence that implied a sizing effect, leaving a correct statement
of a real bias in the forecaster's calibration with no bearing on the position it takes.

None of these corrections altered the central result, under which the seven models remain close to
indistinguishable and the surviving separations are small and conditional. Section 13 of `SPEC.md`
documents each correction in full.

## Open items

The results, tables, and code are complete and reproducible, while the presentation of the paper and
the choice of venue remain unsettled, and the paragraphs below state the position on each.

**The paper.** The paper remains in preparation and is not yet part of this repository. The reading
copy I produced through a document converter and a browser, after the local TeX installation proved
incomplete, reads as an unfinished draft. A proper typeset build is the next step, and a short trial
build of one methods section and one results section, read alongside the full manuscript, will
determine whether the remaining work is typesetting or a further revision of the prose. The paper
will join the repository once complete, and until then this log and Section 13 of `SPEC.md` carry its
argument and its qualifications.

**The venue.** Holding the paper back until the typeset build is complete costs the repository
nothing, since the code, the decision log, the result tables, and the gallery already stand on their
own. Two venues remain under consideration once the paper is ready. SSRN accepts a PDF directly and
requires no endorsement, whereas arXiv offers a stronger imprint but requires both the LaTeX build
and a first-time-submitter endorsement, and neither conflicts with a public code repository. No
preprint has yet been posted.
