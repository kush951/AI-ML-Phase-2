"""
PlaceMux PDF Report Generator
Generates comprehensive fairness audit and model performance reports
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import json
from datetime import datetime
import os

class PlaceMuxReportGenerator:
    def __init__(self, output_filename='PlaceMux_Fairness_Report.pdf'):
        self.filename = output_filename
        self.doc = SimpleDocTemplate(output_filename, pagesize=letter,
                                    rightMargin=0.5*inch, leftMargin=0.5*inch,
                                    topMargin=0.75*inch, bottomMargin=0.75*inch)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        
    def _setup_styles(self):
        """Setup custom styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold',
            borderColor=colors.HexColor('#667eea'),
            borderWidth=2,
            borderPadding=8,
            borderBottomWidth=3,
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14,
        ))
        
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            fontName='Helvetica-Bold',
        ))
    
    def add_title_page(self):
        """Add title page"""
        self.story.append(Spacer(1, 0.8*inch))
        
        # Title
        title = Paragraph("🎯 PlaceMux", self.styles['CustomTitle'])
        self.story.append(title)
        
        # Subtitle
        subtitle = Paragraph(
            "AI-Powered Student-Job Matching<br/>Fairness Audit & Model Performance Report",
            ParagraphStyle('Subtitle', parent=self.styles['Normal'], 
                         fontSize=14, textColor=colors.HexColor('#666'),
                         alignment=TA_CENTER, spaceAfter=12)
        )
        self.story.append(subtitle)
        
        self.story.append(Spacer(1, 0.4*inch))
        
        # Metadata
        metadata_style = ParagraphStyle('Metadata', parent=self.styles['Normal'],
                                       fontSize=10, textColor=colors.HexColor('#888'),
                                       alignment=TA_CENTER)
        metadata = Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}<br/>"
            f"Version: 1.0 | Phase 3 | Week 6",
            metadata_style
        )
        self.story.append(metadata)
        
        self.story.append(Spacer(1, 0.5*inch))
        
        # Executive Summary Box
        exec_summary = Paragraph(
            "<b>Executive Summary</b><br/><br/>"
            "This report presents a comprehensive fairness audit and machine learning model "
            "performance analysis for PlaceMux, an AI-powered student-job matching platform. "
            "The analysis covers model accuracy, fairness metrics across demographic groups, "
            "and recommendations for deployment.",
            self.styles['CustomBody']
        )
        self.story.append(exec_summary)
        
        self.story.append(PageBreak())
    
    def add_table_of_contents(self):
        """Add table of contents"""
        toc_title = Paragraph("Table of Contents", self.styles['CustomHeading'])
        self.story.append(toc_title)
        
        toc_items = [
            "1. Executive Summary",
            "2. Model Performance Metrics",
            "3. Fairness Audit Results",
            "4. Demographic Analysis",
            "5. Recommendations & Conclusions",
            "6. Technical Specifications",
        ]
        
        for item in toc_items:
            p = Paragraph(item, self.styles['CustomBody'])
            self.story.append(p)
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_model_metrics_section(self, metrics):
        """Add model metrics section"""
        self.story.append(Paragraph("2. Model Performance Metrics", self.styles['CustomHeading']))
        
        intro = Paragraph(
            "The following metrics were computed on the held-out test set to evaluate "
            "model performance on unseen data. These metrics are crucial for understanding "
            "the reliability and effectiveness of the matching algorithm.",
            self.styles['CustomBody']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))
        
        # Metrics table
        metrics_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Accuracy', f"{metrics['accuracy']:.1%}", 'Proportion of correct predictions'],
            ['Precision', f"{metrics['precision']:.1%}", 'Accuracy of positive predictions'],
            ['Recall', f"{metrics['recall']:.1%}", 'Coverage of actual positives'],
            ['F1 Score', f"{metrics['f1']:.1%}", 'Harmonic mean of precision & recall'],
            ['ROC-AUC', f"{metrics['auc']:.1%}", 'Classification performance across thresholds'],
            ['False Positive Rate', f"{metrics['fpr']:.1%}", 'Incorrect positive predictions'],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.8*inch, 1.2*inch, 2.2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
        ]))
        
        self.story.append(metrics_table)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Interpretation
        interpretation = Paragraph(
            f"<b>Interpretation:</b> With an accuracy of {metrics['accuracy']:.1%} and F1 score of {metrics['f1']:.1%}, "
            f"the model demonstrates reliable performance on unseen data. The precision of {metrics['precision']:.1%} "
            f"indicates that when the model recommends a match, it is correct {metrics['precision']:.1%} of the time.",
            self.styles['CustomBody']
        )
        self.story.append(interpretation)
        
        self.story.append(PageBreak())
    
    def add_fairness_section(self, fairness_report):
        """Add fairness audit section"""
        self.story.append(Paragraph("3. Fairness Audit Results", self.styles['CustomHeading']))
        
        if 'fairness_score' in fairness_report:
            score_info = fairness_report['fairness_score']
            
            score_data = [
                ['Fairness Metric', 'Score', 'Status'],
                ['Overall Fairness', f"{score_info['overall_score']:.1f}/100", self._get_status(score_info['overall_score'])],
                ['Disparate Impact', f"{score_info['di_score']:.1f}/100", self._get_status(score_info['di_score'])],
                ['Predictive Parity', f"{score_info['pp_score']:.1f}/100", self._get_status(score_info['pp_score'])],
                ['Equalized Odds', f"{score_info['eo_score']:.1f}/100", self._get_status(score_info['eo_score'])],
            ]
            
            score_table = Table(score_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            
            self.story.append(score_table)
            self.story.append(Spacer(1, 0.2*inch))
            
            # Overall assessment
            assessment = Paragraph(
                f"<b>Overall Assessment:</b> {score_info['status']}",
                ParagraphStyle('Assessment', parent=self.styles['Normal'],
                             fontSize=11, textColor=colors.HexColor('#dc3545'),
                             fontName='Helvetica-Bold')
            )
            self.story.append(assessment)
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Fairness concepts explanation
        concepts = Paragraph(
            "<b>Fairness Concepts:</b><br/>"
            "• <b>Disparate Impact:</b> Checks if recommendation rates differ significantly across groups (4/5 rule)<br/>"
            "• <b>Predictive Parity:</b> Ensures positive prediction rate is consistent across demographics<br/>"
            "• <b>Equalized Odds:</b> Verifies false positive and false negative rates are equal across groups",
            self.styles['CustomBody']
        )
        self.story.append(concepts)
        
        self.story.append(PageBreak())
    
    def add_demographic_analysis(self, fairness_report):
        """Add demographic breakdown"""
        self.story.append(Paragraph("4. Demographic Analysis", self.styles['CustomHeading']))
        
        if 'audit_results' not in fairness_report:
            return
        
        results = fairness_report['audit_results']
        
        for demographic, data in results.items():
            self.story.append(Paragraph(f"<b>{demographic.upper()}</b>", 
                                       self.styles['Heading3']))
            
            if 'group_metrics' in data:
                group_metrics = data['group_metrics']
                
                # Create metrics table
                table_data = [['Group', 'Count', 'Accuracy', 'Recall', 'PPR']]
                
                for group, metrics in group_metrics.items():
                    table_data.append([
                        str(group),
                        str(metrics['count']),
                        f"{metrics['accuracy']:.2%}",
                        f"{metrics['recall']:.2%}",
                        f"{metrics['ppr']:.2%}",
                    ])
                
                if len(table_data) > 1:
                    demo_table = Table(table_data, colWidths=[1.2*inch, 1*inch, 1.2*inch, 1.2*inch, 1*inch])
                    demo_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ]))
                    self.story.append(demo_table)
            
            self.story.append(Spacer(1, 0.15*inch))
        
        self.story.append(PageBreak())
    
    def add_recommendations(self):
        """Add recommendations section"""
        self.story.append(Paragraph("5. Recommendations & Conclusions", self.styles['CustomHeading']))
        
        recommendations = Paragraph(
            "<b>Key Recommendations:</b><br/><br/>"
            "1. <b>Deployment Readiness:</b> With strong model performance and fairness metrics, "
            "the system is ready for pilot deployment in a controlled environment.<br/><br/>"
            
            "2. <b>Continuous Monitoring:</b> Implement real-time monitoring of predictions across demographic groups "
            "to detect any drift in fairness metrics post-deployment.<br/><br/>"
            
            "3. <b>Model Retraining:</b> Establish a quarterly retraining schedule with updated data to maintain "
            "model performance and fairness as the student and job populations evolve.<br/><br/>"
            
            "4. <b>Explainability:</b> Ensure all predictions provided to users include clear explanations of the "
            "matching criteria to build trust and allow for human review.<br/><br/>"
            
            "5. <b>Privacy & Consent:</b> Maintain DPDP compliance by ensuring explicit consent for data usage and "
            "easy data deletion mechanisms for all users.<br/><br/>"
            
            "6. <b>Bias Mitigation:</b> Continue monitoring for emerging biases. Consider collecting additional "
            "demographic data to enable intersectional analysis.",
            self.styles['CustomBody']
        )
        self.story.append(recommendations)
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Conclusion
        conclusion = Paragraph(
            "<b>Conclusion:</b><br/>"
            "PlaceMux demonstrates strong potential as an AI-powered recruitment tool that balances accuracy "
            "with fairness. The comprehensive fairness audit confirms that the system maintains equitable treatment "
            "across demographic groups while delivering reliable match predictions. With continued monitoring and "
            "the recommended improvements, PlaceMux can become a trusted platform for student-job matching.",
            self.styles['CustomBody']
        )
        self.story.append(conclusion)
        
        self.story.append(PageBreak())
    
    def add_technical_specs(self):
        """Add technical specifications"""
        self.story.append(Paragraph("6. Technical Specifications", self.styles['CustomHeading']))
        
        tech_info = Paragraph(
            "<b>Model Architecture:</b><br/>"
            "• Algorithm: Random Forest Classifier (100 estimators)<br/>"
            "• Feature Count: 17 engineered features<br/>"
            "• Training Samples: ~800 (80% of 1000 total)<br/>"
            "• Test Samples: ~200 (20% of 1000 total)<br/><br/>"
            
            "<b>Features Used:</b><br/>"
            "• Student: Years of experience, GPA, projects, internships, skill scores<br/>"
            "• Job: Experience requirements, GPA requirement, salary range, skills<br/>"
            "• Computed: Skill match ratios, compatibility scores<br/><br/>"
            
            "<b>Data Preprocessing:</b><br/>"
            "• Categorical encoding: Label Encoding for 5 categorical features<br/>"
            "• Feature scaling: StandardScaler for model training<br/>"
            "• Stratified train-test split to preserve class distribution<br/><br/>"
            
            "<b>Fairness Assessment Framework:</b><br/>"
            "• Demographic parity (disparate impact) analysis<br/>"
            "• Predictive parity across groups<br/>"
            "• Equalized odds evaluation<br/>"
            "• False positive and false negative rate analysis",
            self.styles['CustomBody']
        )
        self.story.append(tech_info)
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Footer note
        footer_note = Paragraph(
            "<i>This report is auto-generated by PlaceMux Reporting System. "
            "For questions or clarifications, please contact the AI/ML Engineering team.</i>",
            ParagraphStyle('FooterNote', parent=self.styles['Normal'],
                         fontSize=9, textColor=colors.HexColor('#999'),
                         alignment=TA_CENTER)
        )
        self.story.append(footer_note)
    
    def _get_status(self, score):
        """Get status based on score"""
        if score >= 80:
            return "✓ EXCELLENT"
        elif score >= 70:
            return "✓ GOOD"
        elif score >= 60:
            return "⚠ FAIR"
        else:
            return "✗ POOR"
    
    def generate(self, metrics=None, fairness_report=None):
        """Generate complete report"""
        self.add_title_page()
        self.add_table_of_contents()
        
        if metrics:
            self.add_model_metrics_section(metrics)
        
        if fairness_report:
            self.add_fairness_section(fairness_report)
            self.add_demographic_analysis(fairness_report)
        
        self.add_recommendations()
        self.add_technical_specs()
        
        # Build PDF
        self.doc.build(self.story)
        print(f"\n✓ Report generated: {self.filename}")
        return self.filename


def generate_sample_report():
    """Generate sample report with mock data"""
    
    # Sample metrics
    metrics = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.89,
        'f1': 0.87,
        'auc': 0.91,
        'fpr': 0.08,
    }
    
    # Sample fairness report
    fairness_report = {
        'fairness_score': {
            'overall_score': 82.5,
            'di_score': 85.0,
            'pp_score': 80.0,
            'eo_score': 82.0,
            'status': '✓ GOOD - Monitor in production',
        },
        'audit_results': {
            'gender': {
                'group_metrics': {
                    'M': {'count': 600, 'accuracy': 0.87, 'recall': 0.88, 'ppr': 0.42},
                    'F': {'count': 400, 'accuracy': 0.87, 'recall': 0.91, 'ppr': 0.44},
                }
            },
            'region': {
                'group_metrics': {
                    'Metro': {'count': 400, 'accuracy': 0.88, 'recall': 0.90, 'ppr': 0.43},
                    'Tier1': {'count': 300, 'accuracy': 0.87, 'recall': 0.88, 'ppr': 0.42},
                    'Tier2': {'count': 200, 'accuracy': 0.85, 'recall': 0.87, 'ppr': 0.41},
                    'Tier3': {'count': 100, 'accuracy': 0.83, 'recall': 0.85, 'ppr': 0.40},
                }
            }
        }
    }
    
    generator = PlaceMuxReportGenerator('PlaceMux_Fairness_Report.pdf')
    generator.generate(metrics, fairness_report)
    
    print("\n" + "="*60)
    print("SAMPLE REPORT GENERATED")
    print("="*60)


if __name__ == "__main__":
    generate_sample_report()
