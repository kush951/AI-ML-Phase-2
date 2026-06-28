"""
PlaceMux PDF Report Generator
Generates comprehensive evaluation and metrics report
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PlaceMuxReportGenerator:
    """Generate comprehensive PDF report for PlaceMux Recommendation System"""

    def __init__(self, output_path: str = "placemux_report.pdf"):

        self.output_path = output_path

        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        self.styles = getSampleStyleSheet()

        # FIXED STYLE SETUP
        self._setup_custom_styles()

        self.story = []

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""

        # -------------------------------------------------
        # MODIFY EXISTING BodyText INSTEAD OF ADDING AGAIN
        # -------------------------------------------------
        body_style = self.styles['BodyText']
        body_style.fontSize = 10
        body_style.alignment = TA_JUSTIFY
        body_style.spaceAfter = 12
        body_style.leading = 14

        # -------------------------------------------------
        # CUSTOM TITLE
        # -------------------------------------------------
        if 'CustomTitle' not in self.styles:

            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#667eea'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))

        # -------------------------------------------------
        # SECTION HEADING
        # -------------------------------------------------
        if 'SectionHeading' not in self.styles:

            self.styles.add(ParagraphStyle(
                name='SectionHeading',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#667eea'),
                spaceAfter=12,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ))

        # -------------------------------------------------
        # SUB HEADING
        # -------------------------------------------------
        if 'SubHeading' not in self.styles:

            self.styles.add(ParagraphStyle(
                name='SubHeading',
                parent=self.styles['Heading3'],
                fontSize=11,
                textColor=colors.HexColor('#764ba2'),
                spaceAfter=10,
                fontName='Helvetica-Bold'
            ))

        # -------------------------------------------------
        # SUBTITLE STYLE
        # -------------------------------------------------
        if 'SubtitleStyle' not in self.styles:

            self.styles.add(ParagraphStyle(
                name='SubtitleStyle',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#764ba2'),
                alignment=TA_CENTER,
                spaceAfter=20
            ))

        # -------------------------------------------------
        # INFO STYLE
        # -------------------------------------------------
        if 'InfoStyle' not in self.styles:

            self.styles.add(ParagraphStyle(
                name='InfoStyle',
                parent=self.styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#666666'),
                alignment=TA_CENTER,
                spaceAfter=6
            ))

    def add_title_page(self, title: str, subtitle: str):

        date = datetime.now().strftime("%B %d, %Y")

        self.story.append(Spacer(1, 2 * inch))

        self.story.append(
            Paragraph(title, self.styles['CustomTitle'])
        )

        self.story.append(Spacer(1, 0.2 * inch))

        self.story.append(
            Paragraph(subtitle, self.styles['SubtitleStyle'])
        )

        self.story.append(Spacer(1, 1 * inch))

        self.story.append(
            Paragraph(
                f"<b>Report Date:</b> {date}",
                self.styles['InfoStyle']
            )
        )

        self.story.append(
            Paragraph(
                "<b>System Version:</b> PlaceMux v1.0",
                self.styles['InfoStyle']
            )
        )

        self.story.append(
            Paragraph(
                "<b>Phase:</b> Phase 2 Industry Immersion",
                self.styles['InfoStyle']
            )
        )

        self.story.append(PageBreak())

    def add_section(self, heading: str, content: str):

        self.story.append(
            Paragraph(heading, self.styles['SectionHeading'])
        )

        self.story.append(Spacer(1, 0.15 * inch))

        self.story.append(
            Paragraph(content, self.styles['BodyText'])
        )

        self.story.append(Spacer(1, 0.3 * inch))

    def add_model_comparison_table(self, model_scores: dict):

        self.story.append(
            Paragraph(
                "Model Performance Comparison",
                self.styles['SectionHeading']
            )
        )

        self.story.append(Spacer(1, 0.15 * inch))

        table_data = [[
            'Model',
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'ROC-AUC'
        ]]

        for model_name, scores in model_scores.items():

            metrics = scores['test']

            table_data.append([
                model_name.replace('_', ' ').title(),
                f"{metrics['accuracy']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1_score']:.3f}",
                f"{metrics['roc_auc']:.3f}"
            ])

        table = Table(table_data)

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor('#f5f5f5')
            ])
        ]))

        self.story.append(table)

        self.story.append(Spacer(1, 0.3 * inch))

    def generate(self, model_scores: dict):

        logger.info(f"Generating PDF report: {self.output_path}")

        # -------------------------------------------------
        # TITLE PAGE
        # -------------------------------------------------
        self.add_title_page(
            "PlaceMux Recommendation System v1",
            "Comprehensive Evaluation Report"
        )

        # -------------------------------------------------
        # EXECUTIVE SUMMARY
        # -------------------------------------------------
        executive_summary = """
        This report documents the development, training,
        and evaluation of the PlaceMux Recommendation System.
        The system successfully achieves strong recommendation
        quality using machine learning models with explainability.
        """

        self.add_section(
            "Executive Summary",
            executive_summary
        )

        # -------------------------------------------------
        # SYSTEM OVERVIEW
        # -------------------------------------------------
        overview = """
        PlaceMux Recommendation System v1 is an AI-powered
        placement recommendation platform designed to match
        students with jobs using machine learning algorithms.
        """

        self.add_section(
            "System Overview",
            overview
        )

        # -------------------------------------------------
        # METHODOLOGY
        # -------------------------------------------------
        methodology = """
        Multiple ML models were trained and evaluated:
        Baseline, Logistic Regression, Random Forest,
        and Gradient Boosting.
        """

        self.add_section(
            "Methodology",
            methodology
        )

        # -------------------------------------------------
        # TABLE
        # -------------------------------------------------
        self.add_model_comparison_table(model_scores)

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------
        results = """
        Gradient Boosting achieved the best F1-score and
        was selected as the production-ready model.
        """

        self.add_section(
            "Results & Findings",
            results
        )

        # -------------------------------------------------
        # CONCLUSION
        # -------------------------------------------------
        conclusion = """
        The system is production-ready and capable of generating
        explainable recommendations for placement officers.
        """

        self.add_section(
            "Conclusion",
            conclusion
        )

        # -------------------------------------------------
        # BUILD PDF
        # -------------------------------------------------
        self.doc.build(self.story)

        logger.info(
            f"PDF report generated successfully: {self.output_path}"
        )

        return self.output_path


def generate_sample_report(
    model_scores: dict,
    output_path: str = "placemux_report.pdf"
):

    generator = PlaceMuxReportGenerator(output_path)

    return generator.generate(model_scores)


if __name__ == "__main__":

    sample_scores = {

        'baseline': {
            'test': {
                'accuracy': 0.72,
                'precision': 0.75,
                'recall': 0.70,
                'f1_score': 0.72,
                'roc_auc': 0.75
            }
        },

        'logistic': {
            'test': {
                'accuracy': 0.81,
                'precision': 0.83,
                'recall': 0.80,
                'f1_score': 0.81,
                'roc_auc': 0.88
            }
        },

        'random_forest': {
            'test': {
                'accuracy': 0.85,
                'precision': 0.87,
                'recall': 0.84,
                'f1_score': 0.85,
                'roc_auc': 0.91
            }
        },

        'gradient_boost': {
            'test': {
                'accuracy': 0.88,
                'precision': 0.92,
                'recall': 0.87,
                'f1_score': 0.89,
                'roc_auc': 0.94
            }
        }
    }

    generate_sample_report(sample_scores)