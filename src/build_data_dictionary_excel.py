"""Build the Dunnhumby data dictionary used by the propensity project."""
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'DUNNHUMBY_DATA_DICTIONARY.xlsx'
GUIDE = 'Dunnhumby The Complete Journey User Guide, trang '

# Tên cột, ý nghĩa, giá trị/ví dụ, vai trò, giới hạn, nguồn.
tables = {
'transaction_data': ("Một dòng sản phẩm trong một basket của household.", [
('household_key','Mã định danh household','Ví dụ: 2375','Nối bảng và nhóm lịch sử theo household','Mã định danh, không phải đại lượng đo', '3'),
('BASKET_ID','Mã một lần mua','Ví dụ: 26984851472','Đếm basket và tạo target','Một basket có thể gồm nhiều dòng sản phẩm','3'),
('DAY','Chỉ số ngày giao dịch','1–711 trong bảng giao dịch','Xác định cutoff, lịch sử và cửa sổ nhãn','Chưa xác minh ánh xạ sang ngày lịch','3'),
('PRODUCT_ID','Mã sản phẩm','Ví dụ: 1004906','Nối product, coupon và causal_data','Mã định danh','3'),
('QUANTITY','Số lượng sản phẩm mua trong chuyến','Ví dụ: 1','Mô tả lượng mua khi hiểu rõ đơn vị','Chưa xác minh quy ước cho mọi department; không cộng xuyên nhóm','3'),
('SALES_VALUE','Số tiền retailer nhận từ dòng sản phẩm','USD; ví dụ 1.39','Mô tả giá trị giao dịch và basket','Có thể khác số tiền khách thực trả khi có manufacturer coupon','3'),
('STORE_ID','Mã cửa hàng','Ví dụ: 364','Nối causal_data, khảo sát cửa hàng','Không suy ra vị trí từ mã','3'),
('RETAIL_DISC','Điều chỉnh do chương trình loyalty của retailer','Thường âm hoặc 0; có giá trị dương','Mô tả discount; EDA đánh dấu giá trị âm','Không giả định mọi giá trị đều âm','3; dữ liệu thực tế'),
('TRANS_TIME','Thời điểm giao dịch trong ngày','HHMM; 1631 là 16:31','Mô tả thời điểm mua','Không cung cấp ngày lịch','3'),
('WEEK_NO','Tuần giao dịch theo chỉ số','1–102','Nối causal_data và phân tích theo tuần','Không phải số tuần đã xác minh của năm lịch','3'),
('COUPON_DISC','Giảm giá do coupon nhà sản xuất','Thông thường âm hoặc 0','Đánh dấu coupon discount trong giao dịch','Không chứng minh tác động tăng thêm','3'),
('COUPON_MATCH_DISC','Khoản retailer đối ứng coupon nhà sản xuất','Thông thường âm hoặc 0','Mô tả discount bổ sung','Không chứng minh việc tiếp xúc hay tác động nhân quả','3'),
]),
'product': ('Một dòng mô tả một PRODUCT_ID.', [
('PRODUCT_ID','Mã sản phẩm','Ví dụ: 25671','Khóa nối transaction, coupon, causal_data','Mã định danh','5'),
('MANUFACTURER','Mã nhà sản xuất','Ví dụ: 2','Khảo sát các sản phẩm cùng nhà sản xuất','Không phải tên doanh nghiệp','5'),
('DEPARTMENT','Nhóm hàng cấp cao','GROCERY, PASTRY, KIOSK-GAS','Mô tả thành phần basket và độ đa dạng','GROCERY không bao gồm mọi loại thực phẩm','5'),
('BRAND','Loại nhãn hàng','National, Private','Phân tích nhãn quốc gia và nhãn riêng','Không phải tên thương hiệu cụ thể','5'),
('COMMODITY_DESC','Nhóm sản phẩm dưới department','Ví dụ: BREAD','Mô tả loại hàng','Tên nhóm có thể tổng quát hoặc thiếu mô tả','5'),
('SUB_COMMODITY_DESC','Nhóm sản phẩm chi tiết hơn','Ví dụ: BREAD:ITALIAN/FRENCH','Mô tả nhóm hàng chi tiết','Nhóm nhỏ có thể ít quan sát','5'),
('CURR_SIZE_OF_PRODUCT','Kích cỡ bao bì nếu có','Ví dụ: 22 LB; có giá trị trống','Cung cấp bối cảnh quy cách','Không mặc định là đơn vị của QUANTITY','5'),
]),
'hh_demographic': ('Thông tin household được phân loại; 801/2.500 hộ có hồ sơ.', [
('classification_1','Nhãn phân loại dạng nhóm tuổi','Age Group1–Age Group6','Mô tả nhóm household','Không có khoảng tuổi cụ thể được xác nhận','4; dữ liệu thực tế'),
('classification_2','Phân loại household','X, Y, Z','Mô tả nhóm household','Chưa xác minh ý nghĩa từng mã','4; dữ liệu thực tế'),
('classification_3','Phân loại household có thứ tự','Level1–Level12','Mô tả nhóm household','Chưa xác minh thuộc tính và khoảng cụ thể','4; dữ liệu thực tế'),
('HOMEOWNER_DESC','Phân loại tình trạng sở hữu nhà','Homeowner, Probable Owner, Renter, Probable Renter, Unknown','Mô tả household','Probable mang tính ước đoán; thời điểm sẵn có chưa rõ','Dữ liệu thực tế'),
('classification_5','Phân loại household có thứ tự','Group1–Group6','Mô tả nhóm household','Chưa xác minh ý nghĩa cụ thể','4; dữ liệu thực tế'),
('classification_4','Phân loại household có thứ tự','1, 2, 3, 4, 5+','Mô tả nhóm household','Chưa xác minh chính xác thuộc tính','4; dữ liệu thực tế'),
('KID_CATEGORY_DESC','Phân loại liên quan đến trẻ em','1, 2, 3+, None/Unknown','Mô tả thành phần household','None/Unknown gộp không có và không rõ','Dữ liệu thực tế'),
('household_key','Mã household','Ví dụ: 1','Nối với giao dịch','Giữ cả household không có hồ sơ; cần xác minh thời điểm sẵn có','4'),
]),
'campaign_table': ('Một quan hệ household–campaign được ghi nhận.', [
('DESCRIPTION','Loại campaign','TypeA, TypeB, TypeC','Phân loại chiến dịch','Không tự mô tả nội dung','4'),
('household_key','Mã household','Ví dụ: 17','Nối household','Có campaign không chứng minh household đã đọc','4'),
('CAMPAIGN','Mã campaign','1–30 theo guide','Nối campaign_desc và coupon','Mã định danh','4'),
]),
'campaign_desc': ('Thông tin loại và thời gian campaign.', [
('DESCRIPTION','Loại campaign','TypeA, TypeB, TypeC','Phân loại campaign','Không tự mô tả nội dung','4'),
('CAMPAIGN','Mã campaign','Ví dụ: 24','Nối các bảng campaign','Mã định danh','4'),
('START_DAY','Chỉ số ngày bắt đầu campaign','Ví dụ: 659','Đối chiếu thời điểm cutoff','Không có ánh xạ lịch thực được xác nhận','4'),
('END_DAY','Chỉ số ngày kết thúc campaign','Ví dụ: 719','Xem thời gian hiệu lực','Có thể vượt phạm vi DAY của transaction','4'),
]),
'coupon': ('Một coupon có thể áp dụng cho nhiều sản phẩm trong campaign.', [
('COUPON_UPC','Mã coupon','Ví dụ: 10000089061','Nối coupon_redempt','Kiểm tra cùng campaign khi nối','5'),
('PRODUCT_ID','Mã sản phẩm có thể dùng coupon','Ví dụ: 27160','Nối product','Không chứng minh household đã mua sản phẩm','5'),
('CAMPAIGN','Mã campaign','Ví dụ: 4','Nối campaign','TypeA: tập coupon có thể gửi, không xác định chính xác từng coupon hộ đã nhận','5'),
]),
'coupon_redempt': ('Các sự kiện coupon được household sử dụng.', [
('household_key','Mã household','Ví dụ: 1','Nối household','Không có redemption không đồng nghĩa chưa nhận coupon','5'),
('DAY','Chỉ số ngày sử dụng coupon','Ví dụ: 421','Giới hạn thông tin đến cutoff','Không có ánh xạ lịch thực được xác nhận','5'),
('COUPON_UPC','Mã coupon đã sử dụng','Ví dụ: 10000085364','Nối coupon','Dùng cùng campaign khi cần xác định coupon','5'),
('CAMPAIGN','Mã campaign','Ví dụ: 8','Nối coupon và campaign','Mã định danh','5'),
]),
'causal_data': ('Thông tin product–store–week về trưng bày và mailer.', [
('PRODUCT_ID','Mã sản phẩm','Ví dụ: 26190','Nối product và transaction','Cấp sản phẩm, không phải exposure của household','6'),
('STORE_ID','Mã cửa hàng','Ví dụ: 286','Nối theo cửa hàng','Không xác nhận household đã thấy trưng bày','6'),
('WEEK_NO','Chỉ số tuần','Ví dụ: 70','Nối theo tuần','Xác minh thông tin có sẵn tại cutoff','6'),
('display','Vị trí trưng bày','0 không trưng bày; 1 trước cửa hàng; A trên kệ','Mô tả bối cảnh cửa hàng','Các mã khác tra bảng mã; không chứng minh tác động','6'),
('mailer','Vị trí trong mailer','0 không quảng cáo; D trang đầu; F trang cuối','Mô tả bối cảnh quảng cáo','Không xác nhận household đã nhận hoặc xem','6'),
]),
'bien_phan_tich': ('Các biến được tính từ bảng nguồn trong notebook; không phải cột nguồn.', [
('snapshot_day','Ngày chốt lịch sử để dự báo','180–537, mỗi 7 ngày trong EDA train','Xác định cutoff','Một household có thể xuất hiện nhiều snapshot','Notebook EDA'),
('recency_days','Số ngày từ basket gần nhất đến cutoff','Ví dụ: 3','Mô tả mức độ gần đây','Chỉ dùng basket đến cutoff','Notebook EDA'),
('active_days_90d','Số ngày có ít nhất một basket trong (s-90,s]','0–90','Đo tần suất; chọn mẫu lịch hoạt động','Một ngày nhiều basket vẫn tính một ngày','Notebook EDA'),
('fuel_basket_90d','Số basket chứa KIOSK-GAS trong 90 ngày','0 trở lên','Mô tả basket chứa fuel','Không chứng minh fuel gây ra mua tiếp','Notebook EDA'),
('coupon_basket_90d','Số basket có coupon discount hoặc coupon match âm trong 90 ngày','0 trở lên','Mô tả coupon discount trong giao dịch','Khác redemption; không bao gồm riêng retail discount','Notebook EDA'),
('target_any_basket','Có bất kỳ basket trong (s,s+7] hay không','0 hoặc 1','Target chính','Bao gồm fuel, miscellaneous và basket tổng sales bằng 0','Notebook EDA'),
('target_positive_basket','Có basket tổng SALES_VALUE dương trong 7 ngày','0 hoặc 1','So sánh độ nhạy định nghĩa','Không phải target chính','Notebook EDA'),
('target_grocery_department','Có basket chứa GROCERY trong 7 ngày','0 hoặc 1','So sánh độ nhạy định nghĩa','GROCERY không gồm mọi nhóm thực phẩm','Notebook EDA'),
('departments_90d','Số department khác nhau trong (s-90,s]','0 trở lên','Mô tả đa dạng nhóm hàng','Không chứng minh sở thích hoặc nguyên nhân','Notebook EDA'),
('baskets_90d','Số basket trong (s-90,s]','0 trở lên','Mẫu số để tính tỷ lệ discount','Không cộng QUANTITY','Notebook EDA'),
('discount_baskets_90d','Số basket có ít nhất một discount âm trong 90 ngày','0 trở lên','Tử số tỷ lệ discount','Xét RETAIL_DISC, COUPON_DISC hoặc COUPON_MATCH_DISC âm','Notebook EDA'),
('discount_share','discount_baskets_90d / baskets_90d','0–1; thiếu nếu không có basket','Mô tả tỷ lệ basket có discount','Không phải % tiền tiết kiệm hay độ nhạy giá','Notebook EDA'),
('activity_group','Nhóm theo số ngày hoạt động 90 ngày','0, 1–3, 4–9, 10–18, 19+','Phân tầng để so sánh','Nhóm đặt theo quy tắc EDA','Notebook EDA'),
('recency_group','Nhóm theo recency','0–7, 8–14, 15–30, 31–60, 61–90, 91+','Phân tầng để so sánh','Nhóm đặt theo quy tắc EDA','Notebook EDA'),
]),
}

wb = Workbook()
ws = wb.active
ws.title = 'Huong_dan'
for row in [
    ['DATA DICTIONARY — CUSTOMER PURCHASE PROPENSITY'],
    ['Phạm vi', '8 bảng nguồn Dunnhumby và các biến phân tích tiêu biểu trong notebook EDA.'],
    ['Cách đọc', 'Mỗi sheet tương ứng một bảng. Cột “Vai trò” mô tả cách dùng; “Giới hạn” nêu điều chưa thể suy diễn.'],
    ['Target', 'Có ít nhất một basket trong 7 ngày sau cutoff, bao gồm fuel, miscellaneous và zero-sales.'],
    ['Thời gian', 'DAY là chỉ số ngày; chưa xác minh ánh xạ sang ngày lịch thực.'],
    ['Demographic', 'Guide có lỗi hiển thị tên cột trong bảng demographic; không tự gán ý nghĩa chưa được xác nhận.'],
    ['Nguồn', 'Dunnhumby The Complete Journey User Guide, trang 3–6; dữ liệu CSV thực tế; notebook phan_tich_du_lieu_propensity.ipynb.'],
    ['Ghi chú', 'Ví dụ là giá trị quan sát hoặc mã minh họa, không nhất thiết liệt kê đầy đủ miền giá trị.'],
]: ws.append(row)
ws.column_dimensions['A'].width = 24
ws.column_dimensions['B'].width = 115
ws.freeze_panes = 'B2'
for row in ws:
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical='top')
    ws.row_dimensions[row[0].row].height = 34
ws['A1'].font = Font(bold=True, color='FFFFFF', size=14)
ws['A1'].fill = PatternFill('solid', fgColor='174A66')

headers = ['Tên cột', 'Ý nghĩa', 'Giá trị / ví dụ', 'Vai trò', 'Giới hạn / lưu ý', 'Nguồn']
for number, (name, (grain, fields)) in enumerate(tables.items(), 1):
    sheet = wb.create_sheet(name)
    sheet.append([f'{name} — {grain}'])
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=6)
    sheet['A1'].font = Font(bold=True, color='FFFFFF', size=12)
    sheet['A1'].fill = PatternFill('solid', fgColor='174A66')
    sheet['A1'].alignment = Alignment(wrap_text=True, vertical='center')
    sheet.row_dimensions[1].height = 32
    sheet.append(headers)
    for row in fields:
        sheet.append([*row[:5], GUIDE + row[5] if row[5][0].isdigit() else row[5]])
    table = Table(displayName=f'Dictionary{number}', ref=f'A2:F{len(fields)+2}')
    table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
    sheet.add_table(table)
    sheet.freeze_panes = 'B3'
    for i, width in enumerate([25, 48, 40, 44, 58, 28], 1):
        sheet.column_dimensions[get_column_letter(i)].width = width
    for row in sheet.iter_rows(min_row=3):
        sheet.row_dimensions[row[0].row].height = 45
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
    sheet.sheet_view.showGridLines = False

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUTPUT)
check = load_workbook(OUTPUT, read_only=True)
assert len(check.sheetnames) == 10
print(f'Đã tạo {OUTPUT}: {len(check.sheetnames)} sheet, {sum(len(v[1]) for v in tables.values())} định nghĩa.')
