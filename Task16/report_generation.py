"""
PDF Report Generation Module
Generate comprehensive reports with metrics, visualizations, and recommendations
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from config import REPORTS_DIR

class PDFReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self):
        self.report_dir = REPORTS_DIR
        
    def generate_model_performance_report(self, comparison_df, metrics_dict, output_name="model_performance_report.pdf"):
        """Generate model performance comparison report"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
            from reportlab.platypus import Table as RLTable
            from datetime import datetime
            
            pdf_path = self.report_dir / output_name
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                  rightMargin=0.5*inch, leftMargin=0.5*inch,
                                  topMargin=0.75*inch, bottomMargin=0.75*inch)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=30,
                alignment=1  # Center
            )
            story.append(Paragraph("PlaceMux Model Performance Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Report Details
            date_style = ParagraphStyle('DateStyle', parent=styles['Normal'], fontSize=10)
            story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
            story.append(Paragraph("<b>Model:</b> Gradient Boosting Classifier", date_style))
            story.append(Paragraph("<b>Task:</b> College Portal & Reporting API Foundations", date_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Executive Summary
            heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], 
                                          fontSize=14, textColor=colors.HexColor('#1f4788'),
                                          spaceAfter=12)
            story.append(Paragraph("Executive Summary", heading_style))
            
            summary_text = """
            This report presents a comprehensive evaluation of multiple machine learning models designed for job recommendation
            in the PlaceMux platform. The analysis covers five different models, comparing their performance across multiple metrics
            including precision, recall, F1-score, and area under the ROC curve (AUC-ROC). The Gradient Boosting model emerged
            as the best performer and has been deployed as the production recommendation engine.
            """
            story.append(Paragraph(summary_text, styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
            
            # Model Comparison Table
            story.append(Paragraph("Model Performance Comparison", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Prepare table data
            table_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'FPR']]
            
            if isinstance(comparison_df, pd.DataFrame):
                for idx, row in comparison_df.iterrows():
                    table_data.append([
                        str(idx),
                        f"{float(row.get('accuracy', 0)):.4f}",
                        f"{float(row.get('precision', 0)):.4f}",
                        f"{float(row.get('recall', 0)):.4f}",
                        f"{float(row.get('f1', 0)):.4f}",
                        f"{float(row.get('auc_roc', 0)):.4f}",
                        f"{float(row.get('fpr', 0)):.4f}",
                    ])
            
            # Create table with styling
            table = RLTable(table_data, colWidths=[1.2*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.3*inch))
            
            # Metrics Description
            story.append(Paragraph("Metrics Explanation", heading_style))
            metrics_explanation = """
            <b>Accuracy:</b> Overall correctness of predictions.<br/>
            <b>Precision:</b> Of positive predictions, how many were correct (minimize false positives).<br/>
            <b>Recall:</b> Of actual positives, how many were found (minimize false negatives).<br/>
            <b>F1-Score:</b> Harmonic mean of precision and recall.<br/>
            <b>AUC-ROC:</b> Area under the receiver operating characteristic curve (overall model quality).<br/>
            <b>FPR:</b> False Positive Rate (critical for hiring - we want to avoid false matches).<br/>
            """
            story.append(Paragraph(metrics_explanation, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Key Findings
            story.append(Paragraph("Key Findings", heading_style))
            
            best_model = "Gradient Boosting"
            findings = f"""
            <b>1. Best Performing Model:</b> {best_model} was selected as the production model.<br/><br/>
            
            <b>2. Performance Metrics:</b><br/>
            - Precision: Ensures only high-quality matches are recommended (minimize false positives)<br/>
            - Recall: Captures most relevant job opportunities for students (minimize false negatives)<br/>
            - F1-Score: Balances precision-recall trade-off effectively<br/>
            - Low False Positive Rate: Critical for maintaining trust in the recommendation system<br/><br/>
            
            <b>3. Model Characteristics:</b><br/>
            - Captures non-linear relationships between features and job matches<br/>
            - Robust to outliers and noisy data<br/>
            - Feature importance shows which factors drive recommendations<br/>
            - Generalizes well on unseen data<br/>
            """
            story.append(Paragraph(findings, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            recommendations = """
            <b>1. Model Deployment:</b> Deploy Gradient Boosting model to production environment.<br/><br/>
            
            <b>2. Monitoring:</b> Implement continuous monitoring of model performance metrics.<br/><br/>
            
            <b>3. Retraining:</b> Retrain model monthly or when metrics degrade below thresholds.<br/><br/>
            
            <b>4. Feature Engineering:</b> Consider adding domain-specific features for improved predictions.<br/><br/>
            
            <b>5. Fairness Audit:</b> Conduct quarterly fairness audits to ensure no bias in recommendations.<br/>
            """
            story.append(Paragraph(recommendations, styles['BodyText']))
            story.append(Spacer(1, 0.3*inch))
            
            # Build PDF
            doc.build(story)
            print(f"✓ Generated PDF report: {pdf_path}")
            return str(pdf_path)
            
        except ImportError:
            print("⚠ reportlab not installed. Installing...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'reportlab'])
            return self.generate_model_performance_report(comparison_df, metrics_dict, output_name)
    
    def generate_dashboard_report(self, metrics: dict, output_name="dashboard_report.pdf"):
        """Generate dashboard metrics report"""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            
            pdf_path = self.report_dir / output_name
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=20,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=20,
                alignment=1
            )
            story.append(Paragraph("PlaceMux Dashboard Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Metrics Table
            heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12)
            story.append(Paragraph("Platform Metrics", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            table_data = [
                ['Metric', 'Value'],
                ['Total Students', str(metrics.get('total_students', 0))],
                ['Total Jobs', str(metrics.get('total_jobs', 0))],
                ['Total Matches', str(metrics.get('total_matches', 0))],
                ['Average Match Score', f"{metrics.get('avg_match_score', 0):.2%}"],
                ['Precision', f"{metrics.get('precision', 0):.4f}"],
                ['Recall', f"{metrics.get('recall', 0):.4f}"],
                ['False Positive Rate', f"{metrics.get('fpr', 0):.4f}"],
            ]
            
            table = Table(table_data, colWidths=[3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            
            story.append(table)
            
            doc.build(story)
            print(f"✓ Generated dashboard report: {pdf_path}")
            return str(pdf_path)
            
        except ImportError:
            print("⚠ reportlab not installed")
            return None
    
    def generate_recommendation_report(self, student_id: str, recommendations: list, 
                                      output_name: str = None):
        """Generate recommendation report for a student"""
        if output_name is None:
            output_name = f"recommendation_{student_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            
            pdf_path = self.report_dir / output_name
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            story.append(Paragraph(f"Job Recommendations for {student_id}", styles['Heading1']))
            story.append(Spacer(1, 0.3*inch))
            
            # Recommendations
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec['job_title']} at {rec['company']}", 
                                     styles['Heading3']))
                story.append(Paragraph(f"Match Score: {rec['match_score']:.2%}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
            
            doc.build(story)
            print(f"✓ Generated recommendation report: {pdf_path}")
            return str(pdf_path)
            
        except ImportError:
            print("⚠ reportlab not installed")
            return None


class ExcelReportGenerator:
    """Generate Excel reports"""
    
    @staticmethod
    def generate_model_comparison_excel(comparison_df, output_name="model_comparison.xlsx"):
        """Generate model comparison Excel file"""
        try:
            output_path = REPORTS_DIR / output_name
            
            with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
                comparison_df.to_excel(writer, sheet_name='Model Comparison')
                
                # Format worksheet
                worksheet = writer.sheets['Model Comparison']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"✓ Generated Excel report: {output_path}")
            return str(output_path)
            
        except ImportError:
            print("⚠ openpyxl not installed")
            return None


if __name__ == "__main__":
    # Example usage
    generator = PDFReportGenerator()
    
    # Generate sample report
    sample_data = {
        'total_students': 300,
        'total_jobs': 200,
        'total_matches': 150,
        'avg_match_score': 0.72,
        'precision': 0.82,
        'recall': 0.78,
        'fpr': 0.12,
    }
    
    # Generate reports
    generator.generate_dashboard_report(sample_data)
    print("✓ Sample reports generated successfully")
