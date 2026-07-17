"""Shared inspection protocol for TeoTensor estimators."""

from __future__ import annotations

from teotensor.artifacts.types import (
    Diagnostic,
    FigureSpec,
    Observation,
    Report,
    Trace,
)


class ObservabilityMixin:
    """Mixin exposing the TeoTensor inspectability contract.

    Concrete models should override :meth:`report` and :meth:`diagnose`.
    Iterative / visual models also override :meth:`trace` and
    :meth:`visualize`. :meth:`observe` aggregates the available artifacts.
    """

    def report(self) -> Report:
        """Return a structured report for this estimator.

        Returns
        -------
        Report
            Model report.

        Raises
        ------
        NotImplementedError
            If the concrete estimator has not implemented reporting yet.
        """
        class_name = type(self).__name__
        msg = (
            f"{class_name}.report() is not implemented. Override this method "
            "to return a teotensor.artifacts.Report."
        )
        raise NotImplementedError(msg)

    def diagnose(self) -> list[Diagnostic]:
        """Return diagnostic findings for this estimator.

        Returns
        -------
        list of Diagnostic
            Ordered findings (may be empty after a successful override).

        Raises
        ------
        NotImplementedError
            If the concrete estimator has not implemented diagnostics yet.
        """
        class_name = type(self).__name__
        msg = (
            f"{class_name}.diagnose() is not implemented. Override this "
            "method to return a list of teotensor.artifacts.Diagnostic."
        )
        raise NotImplementedError(msg)

    def trace(self) -> list[Trace]:
        """Return iterative traces, if any.

        Returns
        -------
        list of Trace
            Empty by default; iterative models override this.
        """
        return []

    def visualize(self) -> list[FigureSpec]:
        """Return declarative figure specifications, if any.

        Returns
        -------
        list of FigureSpec
            Empty by default. Models must not draw plots here — they only
            describe figures for a renderer.
        """
        return []

    def observe(self) -> Observation:
        """Aggregate report, diagnostics, traces, and figures.

        Returns
        -------
        Observation
            Bundle of available observability artifacts.

        Notes
        -----
        Requires :meth:`report` and :meth:`diagnose` to be implemented by the
        concrete estimator.
        """
        report = self.report()
        diagnostics = tuple(self.diagnose())
        traces = tuple(self.trace())
        figures = tuple(self.visualize())
        # Include report figures that are not already listed.
        if report.figures:
            known = {(fig.kind, fig.title) for fig in figures}
            extra = tuple(
                fig for fig in report.figures if (fig.kind, fig.title) not in known
            )
            figures = figures + extra
        return Observation(
            report=report,
            diagnostics=diagnostics,
            traces=traces,
            figures=figures,
        )
