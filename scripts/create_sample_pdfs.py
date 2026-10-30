"""Generate 5 representative sample Indian Government notification PDFs for testing and indexing."""

import sys
from pathlib import Path
import fitz  # PyMuPDF

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def create_sample_pdfs():
    out_dir = Path("data/sample_pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)

    font_path = "C:/Windows/Fonts/Nirmala.ttc"
    use_font = Path(font_path).exists()

    def add_page_text(doc, text, fontsize=11):
        page = doc.new_page()
        rect = fitz.Rect(50, 50, 550, 750)
        if use_font:
            page.insert_font(fontname="indic", fontfile=font_path)
            page.insert_textbox(rect, text, fontname="indic", fontsize=fontsize)
        else:
            page.insert_textbox(rect, text, fontsize=fontsize)
        return page

    # 1. English PIB Notification
    doc1 = fitz.open()
    add_page_text(doc1, "PRESS INFORMATION BUREAU\nGOVERNMENT OF INDIA\n\nNotification No. PIB/2026/DIG-014\nNew Delhi, dated 15th January 2026\n\nSUBJECT: EXPANSION OF DIGITAL SERVICES UNDER DIGITAL INDIA INITIATIVE\n\nThe Union Cabinet chaired by the Prime Minister has approved the comprehensive expansion\nof public digital service infrastructure across all rural and semi-urban panchayats.\nUnder this notification, citizen services including land records, welfare disbursement, and\neducational certificates shall be provisioned through unified digital service centers.\nState governments are requested to ensure high-speed optical fiber connectivity at each\ndistrict headquarters and block office. Guidelines regarding privacy safeguards and data\nlocalization comply strictly with the Digital Personal Data Protection Act.\nAll departments must submit quarterly compliance reports to the Ministry of Electronics and IT.")
    
    add_page_text(doc1, "SECTION 2: BUDGETARY ALLOCATION AND IMPLEMENTATION TIMELINE\n\nThe Central Government has sanctioned an initial capital outlay of Rs. 4,500 Crores for the\ncurrent financial year. Implementation shall proceed in three distinct phases.\nPhase 1 covers tier-1 and tier-2 districts by June 2026.\nPhase 2 encompasses all aspirational districts by December 2026.\nPhase 3 will complete full coverage across remote border villages by March 2027.\nAuthorized officer: Additional Secretary, Ministry of Electronics and Information Technology.")
    doc1.save(out_dir / "doc1_pib_digital_india_en.pdf")
    doc1.close()

    # 2. Hindi Gazette Notification
    doc2 = fitz.open()
    add_page_text(doc2, "भारत का राजपत्र : असाधारण\nशिक्षा मंत्रालय, भारत सरकार\nअधिसूचना संख्या: शि.मं./२०२६/रा.शि.नी.-४२\nनई दिल्ली, दिनांक: २० फरवरी २०२६\n\nविषय: राष्ट्रीय शिक्षा नीति के अंतर्गत भाषाई समावेशन और डिजिटल शिक्षण का क्रियान्वयन\n\nकेंद्र सरकार द्वारा सर्वसम्मति से निर्णय लिया गया है कि प्राथमिक और माध्यमिक विद्यालयों में\nमातृभाषा और क्षेत्रीय भाषाओं में पठन-पाठन सामग्री को अनिवार्य रूप से उपलब्ध कराया जाएगा।\nइस योजना के अंतर्गत हिंदी, कन्नड़, तमिल, तेलुगु और अन्य संविधान की आठवीं अनुसूची में शामिल\nसभी भाषाओं में डिजिटल पाठ्यपुस्तकें और ई-लर्निंग मॉड्यूल तैयार किए जाएंगे।\nराज्यों के शिक्षा बोर्डों को निर्देश दिया जाता है कि वे अपने पाठ्यक्रम को इस ढांचे के अनुरूप ढालें।")
    
    add_page_text(doc2, "खंड २: शिक्षक प्रशिक्षण एवं मूल्यांकन पद्धति\n\nशिक्षकों के कौशल उन्नयन हेतु विशेष ऑनलाइन कार्यशालाओं का आयोजन प्रत्येक जिला स्तर पर किया जाएगा।\nविद्यार्थियों के सतत एवं समग्र मूल्यांकन के लिए आधुनिक डिजिटल पोर्टल प्रारंभ किया गया है।\nइस पोर्टल पर परीक्षा परिणाम तथा दक्षता रिपोर्ट पारदर्शी ढंग से दर्ज की जाएगी।\nआदेशानुसार: संयुक्त सचिव, स्कूली शिक्षा एवं साक्षरता विभाग, भारत सरकार।")
    doc2.save(out_dir / "doc2_gazette_education_policy_hi.pdf")
    doc2.close()

    # 3. Kannada Rural Development Notification
    doc3 = fitz.open()
    add_page_text(doc3, "ಕರ್ನಾಟಕ ಸರ್ಕಾರ\nಗ್ರಾಮೀಣಾಭಿವೃದ್ಧಿ ಮತ್ತು ಪಂಚಾಯತ್ ರಾಜ್ ಇಲಾಖೆ\nಅಧಿಸೂಚನೆ ಸಂಖ್ಯೆ: ಗ್ರಾಅಪಂ/೨೦೨೬/ಜಲ-೧೮\nಬೆಂಗಳೂರು, ದಿನಾಂಕ: ೧೦ ಮಾರ್ಚ್ ೨೦೨೬\n\nವಿಷಯ: ಗ್ರಾಮೀಣ ಕುಡಿಯುವ ನೀರು ಮತ್ತು ನೈರ್ಮಲ್ಯ ಯೋಜನೆಯ ವಿಸ್ತರಣೆ\n\nರಾಜ್ಯದ ಎಲ್ಲಾ ಗ್ರಾಮ ಪಂಚಾಯತಿಗಳಲ್ಲಿ ಶುದ್ಧ ಕುಡಿಯುವ ನೀರು ಸರಬರಾಜು ಮಾಡಲು ಹೊಸ ಮಾರ್ಗಸೂಚಿಗಳನ್ನು ಹೊರಡಿಸಲಾಗಿದೆ.\nಪ್ರತಿಯೊಂದು ಹಳ್ಳಿಯಲ್ಲಿ ಜಲಜೀವನ್ ಯೋಜನೆಯಡಿ ಮನೆ ಮನೆಗೆ ನಲ್ಲಿ ಸಂಪರ್ಕ ಕಲ್ಪಿಸುವ ಕಾರ್ಯವನ್ನು ನಿಗದಿತ ಅವಧಿಯೊಳಗೆ ಪೂರ್ಣಗೊಳಿಸಬೇಕು.\nಗ್ರಾಮ ಪಂಚಾಯತಿ ಅಭಿವೃದ್ಧಿ ಅಧಿಕಾರಿಗಳು (PDO) ಸ್ಥಳೀಯ ನೀರಿನ ಗುಣಮಟ್ಟ ಪರೀಕ್ಷಾ ವರದಿಗಳನ್ನು ಕಡ್ಡಾಯವಾಗಿ ಸಲ್ಲಿಸಬೇಕು.\nನೀರಿನ ಸಂರಕ್ಷಣೆ ಮತ್ತು ಮಳೆನೀರು ಕೊಯ್ಲು ಪದ್ಧತಿಯನ್ನು ಕಡ್ಡಾಯವಾಗಿ ಜಾರಿಗೆ ತರಲು ಸೂಚಿಸಲಾಗಿದೆ.")
    
    add_page_text(doc3, "ವಿಭಾಗ ೨: ಅನುದಾನ ಹಂಚಿಕೆ ಮತ್ತು ಮೇಲ್ವಿಚಾರಣೆ ಸಮಿತಿ\n\nಪ್ರಸಕ್ತ ಸಾಲಿನ ಬಜೆಟ್‌ನಲ್ಲಿ ಈ ಯೋಜನೆಗಾಗಿ ೧,೨೦೦ ಕೋಟಿ ರೂಪಾಯಿಗಳ ಅನುದಾನವನ್ನು ಮೀಸಲಿರಿಸಲಾಗಿದೆ.\nಜಿಲ್ಲಾ ಮಟ್ಟದಲ್ಲಿ ಜಿಲ್ಲಾಧಿಕಾರಿಗಳ ಅಧ್ಯಕ್ಷತೆಯಲ್ಲಿ ವಿಶೇಷ ಮೇಲ್ವಿಚಾರಣಾ ಸಮಿತಿಯನ್ನು ರಚಿಸಲಾಗಿದೆ.\nಯೋಜನೆಯ ಪ್ರಗತಿಯನ್ನು ಪ್ರತಿ ತಿಂಗಳು ಆನ್‌ಲೈನ್ ಪೋರ್ಟಲ್ ಮೂಲಕ ಪರಿಶೀಲಿಸಲಾಗುವುದು.\nಸರ್ಕಾರದ ಪ್ರಧಾನ ಕಾರ್ಯದರ್ಶಿಗಳು, ಗ್ರಾಮೀಣಾಭಿವೃದ್ಧಿ ಇಲಾಖೆ, ಕರ್ನಾಟಕ ಸರ್ಕಾರ.")
    doc3.save(out_dir / "doc3_karnataka_rural_development_kn.pdf")
    doc3.close()

    # 4. English Finance Notification (with 1 sparse page to test quality gate)
    doc4 = fitz.open()
    add_page_text(doc4, "MINISTRY OF FINANCE\nDEPARTMENT OF REVENUE, CENTRAL BOARD OF DIRECT TAXES\n\nCircular No. 05/2026/IT-REV\nNorth Block, New Delhi, 05th April 2026\n\nSUB: GUIDELINES FOR ELECTRONIC SUBMISSION OF TAX AUDIT REPORTS\n\nIn exercise of powers conferred under the Income-tax Act, the Board hereby issues revised\nprocedures for verification and filing of statutory tax audit representations.\nTaxpayers eligible under Section 44AB must file their declarations through designated API channels.\nFailure to adhere to the prescribed digital timeline shall attract administrative penalties\nas stipulated under the governing regulations.")
    
    # Page 2: Sparse page (just a page number) to test quality gate handling
    p2 = doc4.new_page()
    p2.insert_text((250, 400), "Page 2", fontsize=9)

    # Page 3: Annexure with content
    add_page_text(doc4, "ANNEXURE I: SCHEDULE OF PENAL PROCEDURES\n\nLate filings within 30 days of the deadline incur a compounding assessment of 0.5% per week.\nExemptions are granted only for verified technical outages exceeding 72 continuous hours.\nAll disputes are subject to the jurisdiction of the Appellate Tribunal at New Delhi.")
    doc4.save(out_dir / "doc4_finance_tax_circular_en.pdf")
    doc4.close()

    # 5. Bilingual Health Mission Notification
    doc5 = fitz.open()
    add_page_text(doc5, "MINISTRY OF HEALTH AND FAMILY WELFARE\nGOVERNMENT OF INDIA\n\nAYUSHMAN BHARAT DIGITAL MISSION (ABDM) GUIDELINES\nNotification Ref: MOHFW/2026/ABDM-77\nNirman Bhawan, New Delhi, 12th May 2026\n\nThe National Health Authority has notified updated interoperability standards for\nelectronic health records across all empanelled hospitals and diagnostic clinics.\nPatients may link their Ayushman Bharat Health Account (ABHA) ID to access diagnostic\nreports seamlessly across participating institutions with explicit digital consent.")
    
    add_page_text(doc5, "स्वास्थ्य एवं परिवार कल्याण मंत्रालय\nआयुष्मान भारत डिजिटल मिशन दिशानिर्देश\n\nसभी संबद्ध अस्पतालों और प्रयोगशालाओं को डिजिटल स्वास्थ्य रिकॉर्ड प्रणाली से जोड़ना अनिवार्य है।\nनागरिक अपनी आभा (ABHA) पहचान संख्या का उपयोग करके अपने चिकित्सा रिकॉर्ड को सुरक्षित रूप से देख सकते हैं।\nडेटा सुरक्षा और रोगी की गोपनीयता को सर्वोच्च प्राथमिकता दी गई है।\nअपर सचिव, स्वास्थ्य एवं परिवार कल्याण मंत्रालय, भारत सरकार।")
    doc5.save(out_dir / "doc5_health_mission_hi_en.pdf")
    doc5.close()

    print(f"Successfully generated 5 sample PDFs in {out_dir}:")
    for f in sorted(out_dir.glob("*.pdf")):
        print(f"  - {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    create_sample_pdfs()
