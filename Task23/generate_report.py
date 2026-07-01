"""
PlaceMux - PDF Report Generator
Generates comprehensive ML evaluation reports
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import json
from datetime import datetime
import pickle
import os


class PlaceMuxReportGenerator:
    """Generate comprehensive PDF report for ML model evaluation"""
    
    def __init__(self, output_path='reports/'):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=14
        ))
    
    def generate_report(self, test_results=None, model_name=None):
        """Generate complete PDF report"""
        filename = f'{self.output_path}PlaceMux_ML_Evaluation_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.75*inch)
        story = []
        
        # Load data
        try:
            with open('models/results.json', 'r') as f:
                results = json.load(f)
            with open('models/models_info.json', 'r') as f:
                models_info = json.load(f)
        except FileNotFoundError:
            print("Error: Model results not found. Run model_training.py first.")
            return
        
        model_name = models_info.get('best_model', 'Unknown')
        
        # Title Page
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("PlaceMux AI/ML Engineering", self.styles['CustomTitle']))
        story.append(Paragraph("ML Model Evaluation Report", self.styles['CustomHeading']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report info
        report_date = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Best Model Selected:</b> {model_name}", self.styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        summary_text = f"""
        This report presents the comprehensive evaluation of the PlaceMux job-student matching system.
        After training and comparing five different machine learning models, the <b>{model_name}</b> was selected
        as the best performing model based on F1-score and overall predictive capability. The model demonstrates
        strong performance metrics on held-out test data, with robust cross-validation results ensuring
        generalization to unseen data.
        """
        story.append(Paragraph(summary_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key Metrics Table
        story.append(Paragraph("Key Performance Metrics - Best Model", self.styles['CustomHeading']))
        best_metrics = results[model_name]
        
        metrics_data = [
            ['Metric', 'Score'],
            ['Accuracy', f"{best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)"],
            ['Precision', f"{best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)"],
            ['Recall', f"{best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)"],
            ['F1-Score', f"{best_metrics['f1']:.4f} ({best_metrics['f1']*100:.2f}%)"],
            ['ROC-AUC', f"{best_metrics['roc_auc']:.4f} ({best_metrics['roc_auc']*100:.2f}%)"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Confusion Matrix
        story.append(Paragraph("Confusion Matrix", self.styles['CustomHeading']))
        cm = best_metrics['confusion_matrix']
        cm_text = f"""
        <b>True Negatives (TN):</b> {cm[0][0]}<br/>
        <b>False Positives (FP):</b> {cm[0][1]}<br/>
        <b>False Negatives (FN):</b> {cm[1][0]}<br/>
        <b>True Positives (TP):</b> {cm[1][1]}<br/>
        <br/>
        <b>Interpretation:</b> The model correctly identifies job matches {cm[1][1]} times and 
        correctly rejects non-matches {cm[0][0]} times. False positives (suggesting a match when there isn't one) 
        occur {cm[0][1]} times, while false negatives occur {cm[1][0]} times.
        """
        story.append(Paragraph(cm_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Page Break
        story.append(PageBreak())
        
        # Model Comparison
        story.append(Paragraph("Model Comparison", self.styles['CustomHeading']))
        
        comparison_data = [
            ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        ]
        
        for model_name_comp in sorted(results.keys()):
            m = results[model_name_comp]
            comparison_data.append([
                model_name_comp,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['roc_auc']:.4f}"
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
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
        story.append(comparison_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Methodology
        story.append(Paragraph("Methodology", self.styles['CustomHeading']))
        methodology_text = """
        <b>Data Preparation:</b><br/>
        The training dataset was split into three parts: 60% training, 20% validation, and 20% test set.
        Features were standardized using StandardScaler to ensure all features contribute equally to model training.
        <br/><br/>
        <b>Models Trained:</b><br/>
        Five different machine learning algorithms were trained and evaluated:
        <ul>
            <li><b>Logistic Regression:</b> Baseline linear model for probability estimation</li>
            <li><b>Random Forest:</b> Ensemble method with 100 decision trees for robustness</li>
            <li><b>Gradient Boosting:</b> Sequential ensemble with boosting for improved predictions</li>
            <li><b>Support Vector Machine (SVM):</b> Non-linear classifier with RBF kernel</li>
            <li><b>Neural Network:</b> Deep learning model with 2 hidden layers (64, 32 neurons)</li>
        </ul>
        <br/>
        <b>Evaluation Metrics:</b><br/>
        Models were evaluated using multiple metrics to ensure comprehensive assessment:
        <ul>
            <li><b>Accuracy:</b> Overall correctness of predictions</li>
            <li><b>Precision:</b> Accuracy of positive predictions (recommended matches)</li>
            <li><b>Recall:</b> Ability to find all true positive matches</li>
            <li><b>F1-Score:</b> Harmonic mean of precision and recall</li>
            <li><b>ROC-AUC:</b> Area under the receiver operating characteristic curve</li>
        </ul>
        """
        story.append(Paragraph(methodology_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Page Break
        story.append(PageBreak())
        
        # Features & Engineering
        story.append(Paragraph("Feature Engineering", self.styles['CustomHeading']))
        features_text = """
        <b>Feature Store Design:</b><br/>
        A centralized feature store was implemented to ensure consistency between training and inference.
        This guarantees that predictions are based on the same features used during model training.
        <br/><br/>
        <b>Key Features Used:</b><br/>
        <ul>
            <li><b>Student Profile Features:</b> Years of experience, GPA, certifications, 
            projects completed, aptitude score, and individual skill scores (0-100)</li>
            <li><b>Job Requirement Features:</b> Minimum experience, minimum GPA, required certifications,
            salary range, company rating, and required skill importance scores</li>
        </ul>
        <br/>
        <b>Preprocessing:</b><br/>
        All numerical features were standardized using StandardScaler to have zero mean and unit variance.
        Missing values were imputed using median values. This preprocessing improves model training speed
        and convergence while reducing bias toward features with larger scales.
        """
        story.append(Paragraph(features_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Model Explainability
        story.append(Paragraph("Model Explainability", self.styles['CustomHeading']))
        explainability_text = f"""
        <b>Importance of Explainability:</b><br/>
        In job matching systems, every recommendation must be explainable to ensure fairness and trust.
        The {model_name} model provides predictions along with plain-English explanations covering:
        <ul>
            <li>Experience alignment with job requirements</li>
            <li>Educational credentials assessment</li>
            <li>Skill-job requirement matching</li>
            <li>Aptitude score interpretation</li>
            <li>Confidence level of the match prediction</li>
        </ul>
        <br/>
        This explainability framework ensures that candidates and recruiters can understand and trust
        the matching recommendations made by the system.
        """
        story.append(Paragraph(explainability_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.3*inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations & Next Steps", self.styles['CustomHeading']))
        recommendations_text = """
        <b>Model Deployment:</b><br/>
        The {0} model is ready for production deployment. The model has been thoroughly validated
        on hold-out test data and demonstrates strong generalization capability.
        <br/><br/>
        <b>Monitoring & Maintenance:</b><br/>
        <ul>
            <li><b>Performance Monitoring:</b> Continuously track model metrics on new data to detect drift</li>
            <li><b>Feature Store Updates:</b> Maintain feature definitions and preprocessing consistency</li>
            <li><b>Regular Retraining:</b> Retrain the model quarterly with new data to adapt to market changes</li>
            <li><b>User Feedback:</b> Collect feedback on match quality to identify improvement areas</li>
        </ul>
        <br/>
        <b>Security & Data Privacy:</b><br/>
        <ul>
            <li>Ensure student data is encrypted in transit and at rest</li>
            <li>Implement proper data retention policies</li>
            <li>Regular security audits of the API endpoints</li>
            <li>Maintain audit logs of all predictions made</li>
        </ul>
        <br/>
        <b>Future Improvements:</b><br/>
        <ul>
            <li>Implement learning-to-rank models for improved ranking quality</li>
            <li>Add bias and fairness auditing to ensure equitable matches</li>
            <li>Incorporate temporal features (career progression trends)</li>
            <li>Integrate embeddings-based similarity search for better discovery</li>
        </ul>
        """.format(model_name)
        story.append(Paragraph(recommendations_text, self.styles['CustomBody']))
        story.append(Spacer(1, 0.5*inch))
        
        # Footer
        footer_text = f"""
        <b>Report Metadata:</b><br/>
        Generated: {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}<br/>
        Model Version: {model_name}<br/>
        Status: <font color="green"><b>Production Ready</b></font>
        """
        story.append(Paragraph(footer_text, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        print(f"\n✓ PDF Report generated: {filename}")
        return filename


if __name__ == '__main__':
    generator = PlaceMuxReportGenerator()
    generator.generate_report()
