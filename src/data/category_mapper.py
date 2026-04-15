"""
Unified POI Category Mapper (v2)

Maps SafeGraph/Advan SUB_CATEGORY values to unified categories for cross-country comparison.
Based on US weekly_patterns analysis in notebooks/09_visitor_home_cbgs_coverage.py.

IMPORTANT: Uses actual granular SUB_CATEGORY values from US data (e.g., "Full-Service Restaurants")
not TOP_CATEGORY values (e.g., "Restaurants and Other Eating Places").

Only includes categories with >= 50% visitor_home_cbgs coverage for reliable social mixing analysis.

Usage:
    from src.data.category_mapper import CategoryMapper

    mapper = CategoryMapper()
    df['unified_category'] = df['SUB_CATEGORY'].map(mapper.map_category)

    # Or with DataFrame method (preserves TOP_CATEGORY, SUB_CATEGORY):
    df = mapper.map_dataframe(df)
"""

from typing import Optional
from pathlib import Path


class CategoryMapper:
    """Maps SafeGraph/Advan SUB_CATEGORY to unified categories."""

    # 11 unified categories (cross-country comparable, >= 50% CBGS coverage)
    UNIFIED_CATEGORIES = [
        'food_dining',
        'retail_general',
        'retail_specialty',
        'personal_services',
        'pharmacy_retail',
        'entertainment_recreation',
        'accommodation_travel',
        'financial_services',
        'professional_services',
        'automotive',
        'education_higher',
        'civic_community',
    ]

    # Mapping from SUB_CATEGORY to unified category
    # Only includes categories with >= 50% visitor_home_cbgs coverage
    CATEGORY_MAPPING = {
        # =====================================================================
        # FOOD & DINING (restaurants, grocery, bars, cafes)
        # =====================================================================
        'Full-Service Restaurants': 'food_dining',  # 2.07M, 62.1%
        'Limited-Service Restaurants': 'food_dining',  # 982K, 82.9%
        'Supermarkets and Other Grocery (except Convenience) Stores': 'food_dining',  # 576K, 75.4%
        'Snack and Nonalcoholic Beverage Bars': 'food_dining',  # 393K, 75.3%
        'Drinking Places (Alcoholic Beverages)': 'food_dining',  # 380K, 63.7%
        'Convenience Stores': 'food_dining',  # 242K, 78%
        'Retail Bakeries': 'food_dining',  # 173K, 54.7%
        'Beer, Wine, and Liquor Stores': 'food_dining',  # 109K, 62.4%
        'Cafeterias, Grill Buffets, and Buffets': 'food_dining',  # 45K, 65.3%
        'Breweries': 'food_dining',  # 44K, 73.9%
        'Wineries': 'food_dining',  # 51K, 63.6%
        'Meat Markets': 'food_dining',  # 27K, 61.4%
        'Fish and Seafood Markets': 'food_dining',  # 18K, 79.6%
        'Baked Goods Stores': 'food_dining',  # 11K, 74.4%
        'Fruit and Vegetable Markets': 'food_dining',  # 2K, 66.7%

        # =====================================================================
        # RETAIL - GENERAL (department stores, general merchandise)
        # =====================================================================
        'All Other General Merchandise Stores': 'retail_general',  # 312K, 83.2%
        'Department Stores': 'retail_general',  # 48K, 75.8%

        # =====================================================================
        # RETAIL - SPECIALTY (clothing, electronics, home goods, specialty)
        # =====================================================================
        'All Other Miscellaneous Store Retailers (except Tobacco Stores)': 'retail_specialty',  # 973K, 60%
        'Home Centers': 'retail_specialty',  # 214K, 92.2%
        'Women\'s Clothing Stores': 'retail_specialty',  # 213K, 58.1%
        'Used Merchandise Stores': 'retail_specialty',  # 190K, 57.9%
        'Electronics Stores': 'retail_specialty',  # 184K, 55%
        'Shoe Stores': 'retail_specialty',  # 154K, 74.6%
        'Sporting Goods Stores': 'retail_specialty',  # 135K, 71.7%
        'Family Clothing Stores': 'retail_specialty',  # 131K, 75.8%
        'Furniture Stores': 'retail_specialty',  # 109K, 76.1%
        'Hardware Stores': 'retail_specialty',  # 102K, 67.4%
        'Pet and Pet Supplies Stores': 'retail_specialty',  # 93K, 67%
        'Florists': 'retail_specialty',  # 92K, 50.8%
        'Cosmetics, Beauty Supplies, and Perfume Stores': 'retail_specialty',  # 75K, 62.7%
        'Hobby, Toy, and Game Stores': 'retail_specialty',  # 68K, 60.4%
        'Book Stores': 'retail_specialty',  # 64K, 54.5%
        'Men\'s Clothing Stores': 'retail_specialty',  # 62K, 61.3%
        'Nursery, Garden Center, and Farm Supply Stores': 'retail_specialty',  # 54K, 55.5%
        'Jewelry Stores': 'retail_specialty',  # 43K, 68.7%
        'All Other Home Furnishings Stores': 'retail_specialty',  # 39K, 61.9%
        'Household Appliance Stores': 'retail_specialty',  # 38K, 59.3%
        'Children\'s and Infants\' Clothing Stores': 'retail_specialty',  # 27K, 65.9%
        'Office Supplies and Stationery Stores': 'retail_specialty',  # 25K, 69.1%
        'Tobacco Stores': 'retail_specialty',  # 25K, 63.4%
        'Musical Instrument and Supplies Stores': 'retail_specialty',  # 21K, 62.1%
        'Food (Health) Supplement Stores': 'retail_specialty',  # 21K, 69.8%
        'Outdoor Power Equipment Stores': 'retail_specialty',  # 10K, 61%
        'Window Treatment Stores': 'retail_specialty',  # 9K, 50.5%
        'Luggage and Leather Goods Stores': 'retail_specialty',  # 2K, 62.5%
        'Art Dealers': 'retail_specialty',  # 1K, 50.8%

        # =====================================================================
        # PERSONAL SERVICES (salons, repair, personal care)
        # =====================================================================
        'Beauty Salons': 'personal_services',  # 524K, 73%
        'Nail Salons': 'personal_services',  # 110K, 78.4%
        'Barber Shops': 'personal_services',  # 78K, 75.2%
        'Tax Preparation Services': 'personal_services',  # 163K, 72.3%
        'Coin-Operated Laundries and Drycleaners': 'personal_services',  # 42K, 71.5%
        'Locksmiths': 'personal_services',  # 33K, 50.2%
        'Diet and Weight Reducing Centers': 'personal_services',  # 9K, 63.1%
        'Other Personal Care Services': 'personal_services',  # 5K, 75.5%
        'All Other Personal Services': 'personal_services',  # 5K, 68%
        'Drycleaning and Laundry Services (except Coin-Operated)': 'personal_services',  # 4K, 74.7%
        'Funeral Homes and Funeral Services': 'personal_services',  # 75K, 91.8%
        'Cemeteries and Crematories': 'personal_services',  # 5K, 86.4%

        # =====================================================================
        # PHARMACY & DRUG STORES (pharmacies, optical - NOT healthcare providers)
        # =====================================================================
        'Pharmacies and Drug Stores': 'pharmacy_retail',  # 289K, 82.5%
        'Optical Goods Stores': 'pharmacy_retail',  # 156K, 63.6%
        'All Other Health and Personal Care Stores': 'pharmacy_retail',  # 84K, 59.7%

        # =====================================================================
        # ENTERTAINMENT & RECREATION
        # =====================================================================
        'Fitness and Recreational Sports Centers': 'entertainment_recreation',  # 407K, 76.4%
        'Museums': 'entertainment_recreation',  # 95K, 55.2%
        'Golf Courses and Country Clubs': 'entertainment_recreation',  # 79K, 94.5%
        'Nature Parks and Other Similar Institutions': 'entertainment_recreation',  # 29K, 86.5%
        'Bowling Centers': 'entertainment_recreation',  # 29K, 87.4%
        'Motion Picture Theaters (except Drive-Ins)': 'entertainment_recreation',  # 11K, 96.8%
        'All Other Amusement and Recreation Industries': 'entertainment_recreation',  # 10K, 89.1%
        'Amusement Arcades': 'entertainment_recreation',  # 8K, 86.6%
        'Marinas': 'entertainment_recreation',  # 7K, 87.3%
        'Amusement and Theme Parks': 'entertainment_recreation',  # 5K, 77.8%
        'Racetracks': 'entertainment_recreation',  # 3K, 58%
        'Zoos and Botanical Gardens': 'entertainment_recreation',  # 1K, 88.6%
        'Skiing Facilities': 'entertainment_recreation',  # 1K, 87.5%
        'Drive-In Motion Picture Theaters': 'entertainment_recreation',  # 716, 97.5%

        # =====================================================================
        # ACCOMMODATION & TRAVEL
        # =====================================================================
        'Hotels (except Casino Hotels) and Motels': 'accommodation_travel',  # 520K, 85.7%
        'Travel Agencies': 'accommodation_travel',  # 134K, 54.4%
        'Casino Hotels': 'accommodation_travel',  # 59K, 79.4%
        'RV (Recreational Vehicle) Parks and Campgrounds': 'accommodation_travel',  # 24K, 84.5%
        'Rooming and Boarding Houses, Dormitories, and Workers\' Camps': 'accommodation_travel',  # 16K, 85%
        'All Other Traveler Accommodation': 'accommodation_travel',  # 12K, 63%
        'Bed-and-Breakfast Inns': 'accommodation_travel',  # 3K, 54.1%

        # =====================================================================
        # FINANCIAL SERVICES (banks, insurance, lending)
        # =====================================================================
        'Other Activities Related to Credit Intermediation': 'financial_services',  # 529K, 90.8%
        'Commercial Banking': 'financial_services',  # 351K, 82.6%
        'Direct Property and Casualty Insurance Carriers': 'financial_services',  # 185K, 66.7%
        'Securities Brokerage': 'financial_services',  # 73K, 58.7%
        'Direct Life Insurance Carriers': 'financial_services',  # 53K, 55%
        'Consumer Lending': 'financial_services',  # 42K, 66.8%
        'Insurance Agencies and Brokerages': 'financial_services',  # 28K, 65.9%
        'Investment Advice': 'financial_services',  # 16K, 55.5%
        'Mortgage and Nonmortgage Loan Brokers': 'financial_services',  # 9K, 66%
        'All Other Nondepository Credit Intermediation': 'financial_services',  # 7K, 73.5%
        'Credit Unions': 'financial_services',  # 3K, 74.8%

        # =====================================================================
        # PROFESSIONAL SERVICES (business services, veterinary, printing)
        # =====================================================================
        'Commercial Printing (except Screen and Books)': 'professional_services',  # 240K, 96.4%
        'Veterinary Services': 'professional_services',  # 181K, 71.2%
        'Pet Care (except Veterinary) Services': 'professional_services',  # 157K, 90.7%
        'Convention and Trade Show Organizers': 'professional_services',  # 62K, 68%
        'Offices of Lawyers': 'professional_services',  # 7K, 56.9%
        'Employment Placement Agencies': 'professional_services',  # 9K, 76.6%
        'Temporary Help Services': 'professional_services',  # 5K, 65.9%
        'Advertising Agencies': 'professional_services',  # 751, 60.9%
        'Offices of Certified Public Accountants': 'professional_services',  # 1K, 60%
        'Other Management Consulting Services': 'professional_services',  # 894, 57.4%
        'Landscape Architectural Services': 'professional_services',  # 2K, 65.8%

        # =====================================================================
        # AUTOMOTIVE (gas stations, dealers, repair)
        # =====================================================================
        'Gasoline Stations with Convenience Stores': 'automotive',  # 866K, 91.9%
        'General Automotive Repair': 'automotive',  # 518K, 69.4%
        'Other Gasoline Stations': 'automotive',  # 422K, 63.9%
        'New Car Dealers': 'automotive',  # 319K, 93.3%
        'Automotive Parts and Accessories Stores': 'automotive',  # 303K, 73.1%
        'Tire Dealers': 'automotive',  # 262K, 85.3%
        'Automotive Oil Change and Lubrication Shops': 'automotive',  # 172K, 84.6%
        'Used Car Dealers': 'automotive',  # 154K, 96.8%
        'Car Washes': 'automotive',  # 147K, 66%
        'Other Automotive Mechanical and Electrical Repair and Maintenance': 'automotive',  # 100K, 93.9%
        'Motor Vehicle Towing': 'automotive',  # 98K, 83.8%
        'Truck, Utility Trailer, and RV (Recreational Vehicle) Rental and Leasing': 'automotive',  # 68K, 64.7%
        'Motorcycle, ATV, and All Other Motor Vehicle Dealers': 'automotive',  # 60K, 81.8%
        'Passenger Car Rental': 'automotive',  # 47K, 63.5%
        'Boat Dealers': 'automotive',  # 42K, 77.4%
        'Automotive Glass Replacement Shops': 'automotive',  # 31K, 67.1%
        'Passenger Car Leasing': 'automotive',  # 20K, 59.5%
        'Automotive Body, Paint, and Interior Repair and Maintenance': 'automotive',  # 18K, 71.3%
        'Automotive Transmission Repair': 'automotive',  # 13K, 63.8%
        'All Other Automotive Repair and Maintenance': 'automotive',  # 2K, 69.2%
        'Tire Retreading': 'automotive',  # 3K, 75.2%

        # =====================================================================
        # EDUCATION - HIGHER (colleges, vocational - NOT K-12)
        # =====================================================================
        'Colleges, Universities, and Professional Schools': 'education_higher',  # 62K, 91.1%
        'Junior Colleges': 'education_higher',  # 5K, 90.9%
        'Exam Preparation and Tutoring': 'education_higher',  # 20K, 59.4%
        'Other Technical and Trade Schools': 'education_higher',  # 3K, 64.8%
        'All Other Miscellaneous Schools and Instruction': 'education_higher',  # 5K, 68.5%
        'Cosmetology and Barber Schools': 'education_higher',  # 2K, 89.3%
        'Educational Support Services': 'education_higher',  # 2K, 63.8%
        'Sports and Recreation Instruction': 'education_higher',  # 726, 60.9%

        # =====================================================================
        # CIVIC & COMMUNITY (postal, libraries, associations, religious)
        # =====================================================================
        'Postal Service': 'civic_community',  # 121K, 70.1%
        'Business Associations': 'civic_community',  # 86K, 88%
        'Libraries and Archives': 'civic_community',  # 24K, 81.3%
        'Professional Organizations': 'civic_community',  # 3K, 76.5%
        'National Security': 'civic_community',  # 1K, 73.2%
        'Other Similar Organizations (except Business, Professional, Labor, and Political Organizations)': 'civic_community',  # 5K, 74%
    }

    # Categories to exclude - low CBGS coverage (<50%) or not comparable
    EXCLUDED_SUBCATEGORIES = {
        # Low CBGS coverage (< 50%)
        'Gift, Novelty, and Souvenir Stores',  # 45.4%
        'Floor Covering Stores',  # 47.3%
        'Sewing, Needlework, and Piece Goods Stores',  # 47.7%
        'Clothing Accessories Stores',  # 48.5%
        'All Other Specialty Food Stores',  # 49.3%
        'Other Clothing Stores',  # 49.4%
        'Paint and Wallpaper Stores',  # 49.4%
        'Confectionery and Nut Stores',  # 43.9%
        'Financial Transactions Processing, Reserve, and Clearinghouse Activities',  # 44.3%
        'Historical Sites',  # 20.9%
        'Grape Vineyards',  # 35.8%
        'Other Food Crops Grown Under Cover',  # 17.8%
        'Finfish Farming and Fish Hatcheries',  # 8.6%
        'Political Organizations',  # 16.9%
        'Recreational Goods Rental',  # 34.5%
        'Heating Equipment (except Warm Air Furnaces) Manufacturing',  # 49.5%
        'Sporting and Recreational Goods and Supplies Merchant Wholesalers',  # 49.4%
        'Sewage Treatment Facilities',  # 43.6%
        'Industrial Launderers',  # 46.7%
        'Pottery, Ceramics, and Plumbing Fixture Manufacturing',  # 46.2%
        'Footwear Manufacturing',  # 39.9%
        'Coffee and Tea Manufacturing',  # 48.9%

        # K-12 and childcare (not in US data - excluded for cross-country comparability)
        'Elementary and Secondary Schools',
        'Child Day Care Services',

        # Healthcare providers (not in US data - HIPAA privacy)
        'Offices of Physicians',
        'Offices of Dentists',
        'Offices of Other Health Practitioners',
        'Outpatient Care Centers',
        'General Medical and Surgical Hospitals',
        'Specialty (except Psychiatric and Substance Abuse) Hospitals',
        'Medical and Diagnostic Laboratories',
        'Home Health Care Services',
        'Other Ambulatory Health Care Services',
        'Nursing Care Facilities (Skilled Nursing Facilities)',
        'Continuing Care Retirement Communities and Assisted Living Facilities for the Elderly',
        'Nursing and Residential Care Facilities',

        # Transit infrastructure (use GTFS for transit analysis)
        'Urban Transit Systems',
        'Transit and Ground Passenger Transportation',
        'Interurban and Rural Bus Transportation',
        'Other Transit and Ground Passenger Transportation',
        'Charter Bus Industry',
        'School and Employee Bus Transportation',
        'Rail Transportation',
        'Support Activities for Rail Transportation',
        'Taxi and Limousine Service',
        'Taxi Service',

        # Construction (not public venues)
        'Plumbing, Heating, and Air-Conditioning Contractors',
        'Painting and Wall Covering Contractors',
        'Residential Building Construction',
        'Nonresidential Building Construction',
        'New Multifamily Housing Construction (except For-Sale Builders)',
        'Residential Remodelers',

        # Real estate offices (not visitor destinations)
        'Activities Related to Real Estate',
        'Lessors of Real Estate',
        'Offices of Real Estate Agents and Brokers',
        'Lessors of Nonfinancial Intangible Assets (except Copyrighted Works)',
        'Lessors of Nonresidential Buildings (except Miniwarehouses)',
        'Lessors of Miniwarehouses and Self-Storage Units',
        'Lessors of Residential Buildings and Dwellings',
        'Lessors of Other Real Estate Property',
        'Nonresidential Property Managers',

        # Warehousing/logistics (not public)
        'General Warehousing and Storage',
        'Couriers and Express Delivery Services',
        'Refrigerated Warehousing and Storage',
        'Farm Product Warehousing and Storage',

        # Manufacturing (industrial, not public)
        # (Many manufacturing categories omitted for brevity)
    }

    def __init__(self):
        """Initialize the category mapper."""
        self._category_to_unified = self.CATEGORY_MAPPING.copy()

    def map_category(self, sub_category: str) -> Optional[str]:
        """
        Map a SUB_CATEGORY value to unified category.

        Args:
            sub_category: SafeGraph/Advan SUB_CATEGORY value

        Returns:
            Unified category name, or None if excluded/unmapped
        """
        if sub_category is None:
            return None

        # Strip whitespace
        sub_category = str(sub_category).strip()

        # Check if excluded
        if sub_category in self.EXCLUDED_SUBCATEGORIES:
            return None

        # Look up in mapping
        return self._category_to_unified.get(sub_category, None)

    def map_dataframe(self, df, sub_category_col: str = 'SUB_CATEGORY',
                      top_category_col: str = 'TOP_CATEGORY',
                      output_col: str = 'unified_category',
                      keep_original: bool = True):
        """
        Add unified category column to dataframe.

        IMPORTANT: By default preserves TOP_CATEGORY and SUB_CATEGORY columns
        for downstream analysis. Set keep_original=False to only add unified.

        Args:
            df: DataFrame with SUB_CATEGORY column
            sub_category_col: Name of the SUB_CATEGORY column
            top_category_col: Name of the TOP_CATEGORY column (for reference)
            output_col: Name for the new unified category column
            keep_original: If True, ensures TOP_CATEGORY and SUB_CATEGORY are preserved

        Returns:
            DataFrame with unified_category column added.
            Always preserves: TOP_CATEGORY, SUB_CATEGORY, unified_category
        """
        import pandas as pd

        df = df.copy()

        # Strip whitespace from category columns
        if sub_category_col in df.columns:
            if df[sub_category_col].dtype == 'object':
                df[sub_category_col] = df[sub_category_col].str.strip()

        if top_category_col in df.columns:
            if df[top_category_col].dtype == 'object':
                df[top_category_col] = df[top_category_col].str.strip()

        # Map to unified categories
        df[output_col] = df[sub_category_col].map(self.map_category)

        # Verify original columns are preserved
        if keep_original:
            required_cols = [top_category_col, sub_category_col, output_col]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"Warning: Missing columns that should be preserved: {missing}")

        return df

    def get_unmapped_categories(self, sub_categories) -> set:
        """
        Find SUB_CATEGORY values that don't map to any unified category.

        Args:
            sub_categories: Iterable of SUB_CATEGORY values

        Returns:
            Set of unmapped category names (excluding excluded ones)
        """
        unmapped = set()
        for cat in sub_categories:
            if cat is None:
                continue
            cat = str(cat).strip()
            if cat not in self.EXCLUDED_SUBCATEGORIES and cat not in self._category_to_unified:
                unmapped.add(cat)
        return unmapped

    def is_excluded(self, sub_category: str) -> bool:
        """Check if a category is in the exclusion list."""
        if sub_category is None:
            return False
        return str(sub_category).strip() in self.EXCLUDED_SUBCATEGORIES

    @classmethod
    def get_unified_categories(cls) -> list:
        """Return list of unified category names."""
        return cls.UNIFIED_CATEGORIES.copy()

    @classmethod
    def get_category_description(cls, unified_cat: str) -> str:
        """Return human-readable description for a unified category."""
        descriptions = {
            'food_dining': 'Food & Dining',
            'retail_general': 'General Retail',
            'retail_specialty': 'Specialty Retail',
            'personal_services': 'Personal Services',
            'pharmacy_retail': 'Pharmacy & Drug Stores',
            'entertainment_recreation': 'Entertainment & Recreation',
            'accommodation_travel': 'Accommodation & Travel',
            'financial_services': 'Financial Services',
            'professional_services': 'Professional Services',
            'automotive': 'Automotive',
            'education_higher': 'Higher Education',
            'civic_community': 'Civic & Community',
        }
        return descriptions.get(unified_cat, unified_cat)


def compute_exclusion_stats(df, sub_category_col: str = 'SUB_CATEGORY',
                            visit_col: str = None) -> dict:
    """
    Compute statistics about what gets excluded by the category mapping.

    Args:
        df: DataFrame with SUB_CATEGORY column
        sub_category_col: Name of the category column
        visit_col: Optional column with visit counts (if None, counts rows)

    Returns:
        Dictionary with exclusion statistics
    """
    mapper = CategoryMapper()

    # Clean categories
    categories = df[sub_category_col].str.strip() if df[sub_category_col].dtype == 'object' else df[sub_category_col]

    # Classify each row
    is_mapped = categories.map(lambda x: x in mapper.CATEGORY_MAPPING if x else False)
    is_excluded = categories.map(lambda x: x in mapper.EXCLUDED_SUBCATEGORIES if x else False)
    is_unmapped = ~is_mapped & ~is_excluded & categories.notna()

    if visit_col and visit_col in df.columns:
        # Weight by visits
        total = df[visit_col].sum()
        mapped = df.loc[is_mapped, visit_col].sum()
        excluded = df.loc[is_excluded, visit_col].sum()
        unmapped = df.loc[is_unmapped, visit_col].sum()
        unit = "visits"
    else:
        # Count rows
        total = len(df)
        mapped = is_mapped.sum()
        excluded = is_excluded.sum()
        unmapped = is_unmapped.sum()
        unit = "rows"

    # Breakdown of excluded categories
    excluded_breakdown = {}
    for cat in mapper.EXCLUDED_SUBCATEGORIES:
        mask = categories == cat
        if visit_col and visit_col in df.columns:
            count = df.loc[mask, visit_col].sum()
        else:
            count = mask.sum()
        if count > 0:
            excluded_breakdown[cat] = count

    return {
        'total': total,
        'mapped': mapped,
        'mapped_pct': mapped / total * 100 if total > 0 else 0,
        'excluded': excluded,
        'excluded_pct': excluded / total * 100 if total > 0 else 0,
        'unmapped': unmapped,
        'unmapped_pct': unmapped / total * 100 if total > 0 else 0,
        'excluded_breakdown': dict(sorted(excluded_breakdown.items(), key=lambda x: -x[1])),
        'unit': unit,
    }


def compute_category_distribution(df, sub_category_col: str = 'SUB_CATEGORY',
                                   visit_col: str = None) -> dict:
    """
    Compute visit distribution across unified categories.

    Args:
        df: DataFrame with SUB_CATEGORY column
        sub_category_col: Name of the category column
        visit_col: Optional column with visit counts (if None, counts rows)

    Returns:
        Dictionary mapping unified_category -> count/visits
    """
    mapper = CategoryMapper()

    # Map categories
    df = df.copy()
    df['_unified'] = df[sub_category_col].str.strip().map(mapper.map_category)

    if visit_col and visit_col in df.columns:
        dist = df.groupby('_unified')[visit_col].sum().to_dict()
    else:
        dist = df['_unified'].value_counts().to_dict()

    # Remove None key if present
    dist.pop(None, None)

    return dist


def validate_mapping():
    """Validate the category mapping against loaded data."""
    mapper = CategoryMapper()

    print("Unified POI Category Mapper (v2)")
    print("=" * 50)
    print(f"\nUnified categories: {len(mapper.UNIFIED_CATEGORIES)}")
    for cat in mapper.UNIFIED_CATEGORIES:
        desc = mapper.get_category_description(cat)
        count = sum(1 for v in mapper.CATEGORY_MAPPING.values() if v == cat)
        print(f"  - {desc}: {count} sub-categories")

    print(f"\nTotal mapped SUB_CATEGORY values: {len(mapper.CATEGORY_MAPPING)}")
    print(f"Total excluded SUB_CATEGORY values: {len(mapper.EXCLUDED_SUBCATEGORIES)}")

    # Show key exclusions
    print(f"\nKey excluded categories (cross-country comparability):")
    key_exclusions = [
        ('K-12/Childcare', ['Elementary and Secondary Schools', 'Child Day Care Services']),
        ('Healthcare providers', ['Offices of Physicians', 'Offices of Dentists', 'General Medical and Surgical Hospitals']),
        ('Transit infrastructure', ['Urban Transit Systems', 'Transit and Ground Passenger Transportation']),
        ('Low CBGS coverage', ['Gift, Novelty, and Souvenir Stores', 'Floor Covering Stores', 'Historical Sites']),
    ]
    for group_name, cats in key_exclusions:
        excluded_count = sum(1 for c in cats if c in mapper.EXCLUDED_SUBCATEGORIES)
        print(f"  - {group_name}: {excluded_count} categories excluded")

    return mapper


if __name__ == "__main__":
    validate_mapping()
