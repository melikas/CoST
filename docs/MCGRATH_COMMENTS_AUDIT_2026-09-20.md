# تطبیق نظرات McGrath با پروژهٔ فعلی DSSL

تاریخ بررسی: ۲۰ سپتامبر ۲۰۲۶. نسخهٔ Git هنگام بررسی: `92cd7dc`.

منبع: `C:/Users/umroot/Downloads/Comments_McGrath_2026Aug25.docx`.
متن اصلی Word و وجود بخش‌های comments/footnotes بررسی شد؛ نظرات مرتبط در متن اصلی سند بودند.
این سند، نظرات را به‌عنوان موضوع بررسی در نظر می‌گیرد، نه دستور تغییر خودکار پروژه.

دامنه: کد فعال، تنظیمات HRD/GLOBEM، مقالهٔ `SSL_Rhythmicity`، metadata کش‌های موجود و خروجی‌های `results/{hrd,globem}/narval_v2/tcn_none`.
مقصود از «اجرا شده» وجود پیاده‌سازی و، در صورت نیاز، خروجی واقعی است؛ ذکر یک موضوع در راهنما معادل اجرای تحلیل نیست. نبود شواهد در این مخزن، نبود آن در تمام کارهای پایان‌نامه را اثبات نمی‌کند.

## نتیجهٔ اصلی

پروژه به چند توصیهٔ محاسباتی و ارزیابی پاسخ داده است، اما سه اولویت اصلی استاد هنوز کامل رعایت نشده‌اند: تفکیک ریتم مشاهده‌شده از ریتم درون‌زاد circadian، استفادهٔ دوری از فاز در تمام مسیرها، و تحلیل حساسیت اثر missingness/imputation بر ریتم. بنابراین نمی‌توان گفت نظرات فایل به‌طور کامل اعمال شده‌اند.

## تطبیق پیشنهادهای اصلی، با شماره‌گذاری همان فایل

| بند فایل | درخواست | وضعیت | شواهد و فاصلهٔ باقی‌مانده |
|---|---|---|---|
| ۱ | تمایز endogenous circadian از diurnal/rest-activity/wearable-derived rhythm | اعمال نشده به‌صورت جامع | عنوان مقاله اکنون Biobehavioral Rhythms است، اما مقدمه و نتایج همچنان الگوهای پوشیدنی را circadian معرفی می‌کنند؛ تعریف یکپارچهٔ masking و محدودیت نبود نشانگر درون‌زاد دیده نشد. |
| ۱ | تعریف مفاهیم و محدود کردن تفسیر بالینی | ناقص | محدودیت کلیِ retrospective بودن و اثبات نشدن clinical utility نوشته شده است؛ این جای توضیح اختصاصیِ construct validity و نبود DLMO/مرجع درون‌زاد را نمی‌گیرد. |
| ۱ | در صورت حفظ ادعای circadian برای HR، بررسی ریتم HR پس از حساب کردن حرکت | اجرا نشده | HR و Steps کانال‌های جداگانهٔ ورودی‌اند. تحلیل residual HR پس از تعدیل حرکت در مسیر فعال پیدا نشد. این پیشنهاد مشروط است؛ صرف ورود Steps به شبکه چنین تحلیلی محسوب نمی‌شود. |
| ۲ | برخورد دوری با acrophase در همهٔ مراحل | ناقص، با شکاف مهم | هدف‌های cosinor به صورت cos/sin و خطای فاز دوری هستند و loss پیش‌فرض circular است؛ اما readout پیش‌فرض `angle` است. میانگین‌گیری ویژگی‌ها، baseline شخصی، SD و فاصله بر زاویهٔ خام مانند متغیر خطی اعمال می‌شوند. |
| ۳ | نگهداری تفاوت missingness با inactivity و provenance پرکردن شکاف‌ها | تا حدی اجرا شده | ماسک مشاهدهٔ اصلی، داده در واحد فیزیکی و قواعد جداگانهٔ پاک‌سازی حفظ می‌شوند. در HRD از HR برای رفع ابهام خواب استفاده می‌شود؛ فقدان HR اثبات قطعی non-wear نیست. خروجی مشاهده/عدم‌مشاهده نیز خودبه‌خود علت فقدان را تعیین نمی‌کند. |
| ۳ | گزارش coverage/non-wear برحسب گروه و زمان روز | ناقص | کسری missingness برای هر کانال و قواعد coverage ثبت شده‌اند؛ جدول گروه بالینی × زمان روز و تحلیل علت non-wear در مسیر فعال/خروجی بررسی‌شده دیده نشد. |
| ۳ | مقایسهٔ شاخص‌های ریتم با و بدون imputation و حساسیت به حذف افراد | اجرا نشده | شاخص‌ها صریحاً از completed signals محاسبه می‌شوند. ماسک و cutoff کیفیت وجود دارد، اما تحلیل حساسیت به imputation، طول شکاف یا آستانهٔ حذف افراد اجرا نشده است. |
| ۳ | missingness به‌عنوان predictor مستقل | اجرا نشده | مدل پایهٔ صرفاً مبتنی بر coverage/non-wear یا گزارش مقایسهٔ آن در ladder فعال پیدا نشد. |
| ۴ | توصیف MESOR، دامنه، فاز، regularity و کیفیت برازش در گروه‌ها، با effect size و CI | ناقص | `secondary_endpoint_associations.csv` واقعی وجود دارد و rank-biserial effect size و p اصلاح‌شده با Holm می‌دهد. جدول کامل مقادیر هر گروه، میانگین/پراکندگی دوری فاز، کیفیت fit و CI اثرها ندارد. مقایسهٔ جداگانهٔ phase_cos/phase_sin جانشین کامل جدول توصیفی دوری نیست. |
| ۴ | بررسی شدت پیوستهٔ علائم، هرجا موجود است | در تحلیل اصلی اجرا نشده | ستون‌های CES-D baseline/endpoint در loader HRD نگهداری می‌شوند، اما RQ3 فعال binary endpoint است و تحلیل پیوستهٔ شدت به‌عنوان outcome در خروجی فعال دیده نشد. |
| ۵ | مشخص بودن ابزار، cutoff، زمان سنجش و منشأ depression label | ناقص | کد outcomeهای ثابت endpoint را مصرف می‌کند و گزارش HRD از CES-D≥16 نام می‌برد. بااین‌حال metadata کش‌های واقعی برای هر دو مجموعه `instrument/cutoff provenance pending` دارد؛ منشأ GLOBEM کامل روشن نیست. فایل استاد از BDI-II در Human-Rhythms صحبت می‌کند؛ این تفاوت با گزارش فعلی باید با منبع داده حل و مستند شود. |
| ۵ | ساختن نکردن برچسب هفتگی از دو سنجش و تفکیک trait/state | عمدتاً رعایت شده در طراحی فعلی | RQ3 یک outcome فردی در پایان مطالعه دارد؛ RQ2 label-free است. استفاده از majority برچسب هفتگی به‌جای endpoint در loader GLOBEM صریحاً رد می‌شود. گزارش و future work محدودیت سنجش تغییر طبیعی علائم را می‌پذیرند. اعتبار بالینی تشخیص state change هنوز اثبات نشده است. |
| ۶ | شاخص‌های غیرپارامتری کنار cosinor | بخش اصلی اجرا شده | IS، IV و RA در `individual_markers` محاسبه و در RQ1 و baselineها استفاده شده‌اند. RA به Steps و grid مناسب محدود شده است. L5 و M10 برای محاسبهٔ RA استفاده می‌شوند، اما به‌صورت خروجی مستقل گزارش نمی‌شوند. |
| ۶ | گزارش کیفیت و محدودیت fit تک‌کسینوسی | ناقص | baseline مربوط به Yan مقادیر SNR/RSS/residual SE و اطلاعات fit را تولید می‌کند. این با ارائهٔ کیفیت fit برای شاخص‌های ۲۴ساعتهٔ اصلی، به تفکیک کانال/گروه، یکسان نیست؛ چنین گزارش جامعی دیده نشد. |
| ۶ | محدود کردن تحلیل بر اساس resolution مجموعه‌داده | عمدتاً اجرا شده | HRD با ۱۵ دقیقه و GLOBEM با ۶ ساعت از هم تفکیک شده‌اند. RQ2 مشابه HRD برای GLOBEM N/A و RA استاندارد آن ناموجود است. متن همچنان به دقت بیشتر دربارهٔ Nyquist و امکان/معنای فاز و هارمونیک‌ها نیاز دارد؛ وجود دو باند به معنای دو ریتم با فاز آزاد نیست. |
| ۷ | حفظ و بررسی local time، UTC offset، timezone و DST | فقط محدودیت ثبت شده؛ حل نشده | شروع پنجره و تاریخ حفظ می‌شود، اما metadata کش‌های واقعی `timezone/DST provenance unresolved` است و کد HRD صریحاً اصلاح timezone/DST را حدس نمی‌زند. |
| ۷ | بررسی فصل، نور، برنامهٔ روزانه، chronotype، دارو و متغیرهای بالینی/جمعیتی | اجرا نشده در مسیر فعال | مدل تعدیل‌شده یا تحلیل حساسیت مشخص برای این عوامل پیدا نشد. محدودیت‌های فعلی مقاله نیز همهٔ این عوامل را به‌طور مشخص پوشش نمی‌دهد. تفکیک افراد، این نوع confounding را رفع نمی‌کند. |
| ۸ | baseline شخصی بر اساس گذشتهٔ همان فرد | اجرا شده | چهار هفتهٔ متوالیِ قبل از هفتهٔ جاری، با کنترل ترتیب زمانی و فاصلهٔ شروع پنجره‌ها، استفاده می‌شود؛ افراد آزمون جدا هستند. |
| ۸ | گزارش اینکه دقیقاً چه شاخص قابل‌تفسیری همراه embedding تغییر کرده است | ناقص | RQ2 تغییر فاصلهٔ embedding را با تغییر فاصلهٔ cosinor خام مقایسه می‌کند و timing/strength را جدا می‌سنجد. خروجی اصلی `raw_delta` یک فاصلهٔ خلاصه است؛ جدول کامل Δphase به ساعت، Δamplitude، fit، IS/IV و coverage برای تغییرات طبیعی ارائه نمی‌کند. |
| ۸ | تفکیک تغییر رفتار از internal circadian misalignment | ناقص | ادعای بالینی برای RQ2 مصنوعی محدود شده، اما تعریف صریح رابطهٔ فاز رفتاری در برابر misalignment درون‌زاد و محدودیت نبود مرجع درون‌زاد در متن دیده نشد. sleep midpoint نسبت به activity acrophase هم در تحلیل فعال وجود ندارد. |
| ۹ | گزارش variability و uncertainty به‌جای AUC تنها | عمدتاً اجرا شده | خروجی واقعی RQ3 شامل AUROC_seed_SD و CI بوت‌استرپ paired برای اختلاف مدل‌هاست؛ RQ1/RQ2 نیز CI دارند. CI اختلاف AUC با CI خود AUC یکسان نیست؛ جدول اصلی RQ3 CI مستقیم AUC هر روش را ندارد. عدد 0.876 مربوط به تحلیل قدیمی مورد اشارهٔ استاد را نمی‌توان با جدول HRD فعلی پاسخ‌داده‌شده دانست. |
| ۹ | participant-level separation و tuning روی train | بخش اصلی اجرا شده | `fold_data` افراد test را از pretraining هم حذف می‌کند؛ classifier روی میانگین ویژگی هر فرد است و انتخاب regularization در inner CV افراد train انجام می‌شود. بااین‌حال normalization/imputation ورودی از کل رکورد همان فرد استفاده می‌کند و گذشته‌نگر است؛ تضمین ارزیابی prospective از این طراحی به دست نمی‌آید. |
| ۹ | شواهد مستقیم حفظ ریتم، فراتر از t-SNE/UMAP | اجرا شده | held-out cosinor recovery، خطای دوری فاز، کنترل untrained/raw و آزمون perturbation موجودند و خروجی واقعی دارند. اجرای آزمون، به معنای موفق بودن فرضیه نیست؛ نتایج canonical برخی معیارهای RQ1 را رد می‌کنند. |

## شواهد کلیدی و مسیر بررسی

- اصطلاحات حل‌نشده: [مقدمه](../SSL_Rhythmicity/sections/1-introduction.tex)، به‌ویژه ابتدای متن و ادعاهای circadian در انتها؛ [نتایج](../SSL_Rhythmicity/sections/4-results.tex)، عنوان جدول RQ1 و بخش RQ2. [محدودیت‌ها](../SSL_Rhythmicity/sections/5b-limitations.tex) محدودیت کلی retrospective را دارد، اما توضیح اختصاصی masking/مرجع فاز درون‌زاد را ندارد.
- فاز: [تنظیم HRD](../configs/hrd.json) و [GLOBEM](../configs/globem.json): `phase_readout="angle"` همراه `phase_mode="circular_amp"`. [evaluation_protocol.py](../evaluation_protocol.py): `participant_mean` در خط ۳۳ و مصرف آن در خط ۱۳۴؛ خطای دوری در خط ۵۸۸. [tasks/rhythm.py](../tasks/rhythm.py): `personal_baseline` و `dscore`. loss دوری مشکل میانگین/فاصلهٔ خطی downstream را خودبه‌خود حل نمی‌کند.
- missingness: [hrd_clean.py](../data_processing/hrd_clean.py)، [globem_dataset.py](../data_processing/globem_dataset.py)، [build_cache.py](../scripts/build_cache.py) و [individual_markers](../tasks/rhythm.py). در GLOBEM interpolation درون فرد با extension انتهایی وجود دارد؛ استفاده از ماسک برای eligibility، شاخص را به observed-only تبدیل نمی‌کند.
- تحلیل گروه‌ها: [فایل خروجی HRD](../results/hrd/narval_v2/tcn_none/secondary_endpoint_associations.csv) و کد `summarize` در [evaluation_protocol.py](../evaluation_protocol.py)، بخش Secondary. ستون‌ها عبارت‌اند از marker، channel، تعداد دو گروه، rank_biserial، p و p_holm؛ ستون CI اثر یا میانگین گروه وجود ندارد.
- outcome/timezone: metadata داخل `datasets/cache/hrd_rescue_v1.npz` و `datasets/cache/globem_rescue_v1.npz` خوانده شد. هر دو دارای عبارت‌های unresolved/pending ذکرشده‌اند؛ این صرفاً حدس از README نیست. [build_cache.py](../scripts/build_cache.py) نیز این وضعیت را ثبت می‌کند.
- شدت علائم: [hrd_config.py](../data_processing/hrd_config.py) در `extra_label` نمره‌های CES-D را دارد؛ [evaluation_protocol.py](../evaluation_protocol.py) برای outcome فعال فقط برچسب صفر/یک را می‌پذیرد.
- شاخص‌ها: [tasks/rhythm.py](../tasks/rhythm.py) شامل IS/IV/RA و L5/M10 داخلی است. [tasks/yan_cosinor.py](../tasks/yan_cosinor.py) کیفیت fit را در baseline می‌سازد؛ فازهای آن ابتدا دوری تجمیع می‌شوند، سپس به زمان/زاویه برمی‌گردند و وارد probe معمولی می‌شوند؛ بررسی دوری تمام مسیرها باید این baseline را هم شامل شود.
- شخصی‌سازی: [tasks/personalized.py](../tasks/personalized.py)، چهار هفتهٔ قبل، perturbationهای کنترل‌شده و `raw_delta`/`representation_delta`؛ [future directions](../SSL_Rhythmicity/sections/5c-futuredirections.tex) سنجش‌های مکرر علائم را کار آینده می‌داند.
- آمار اجراشده: [SUMMARY.md مربوط به HRD](../results/hrd/narval_v2/tcn_none/SUMMARY.md)، [RQ3_table.csv](../results/hrd/narval_v2/tcn_none/RQ3_table.csv)، [summary_by_seed.csv](../results/hrd/narval_v2/tcn_none/summary_by_seed.csv) و [rq3_paired_intervals.csv](../results/hrd/narval_v2/tcn_none/rq3_paired_intervals.csv).

## نقاط قوت و موارد بیرون از دامنه

- تحسین جداسازی افراد در Depresjon: خود Depresjon در اجرای canonical فعلی نیست، اما اصل جداسازی فردی و یک سطر ویژگی برای هر فرد در HRD/GLOBEM رعایت می‌شود.
- تحسین سه split تعمیم GLOBEM: اجرای canonical فعلی فقط ۲۰۱۸ را استفاده می‌کند؛ نتایج cross-year و COVID-transition جزو آن نیستند. این مورد «اجرا شده» علامت نمی‌خورد، حتی اگر ابزار یا نتایج تاریخی دیگری در archive وجود داشته باشد.
- تحسین metadata و تمایز measurement/vendor metric: پروژهٔ فعلی نام و نوع کانال، دادهٔ مشتق‌شدهٔ خواب، ماسک مشاهده، نسخهٔ پردازش و hash فایل را نگه می‌دارد. provenance کامل مدل دستگاه، firmware، الگوریتم vendor و نسخهٔ آن برای هر متغیر در قرارداد فعال دیده نشد. زیرساخت TechnoHealth خارج از دامنهٔ این مخزن است.
- تحسین cosinor probe: در پروژهٔ فعلی وجود دارد و اجرا شده؛ موفقیت علمی آن باید از خروجی‌های واقعی نتیجه گرفته شود، نه از وجود کد.
- ingestion چنددستگاهی، alert عدم‌استفاده و جزئیات Embrace Plus/E4 مربوط به pilot/TechnoHealth است؛ از این مخزن نمی‌توان دربارهٔ اعمال آن‌ها حکم داد.

## نکات ویرایشی مرتبط

- عبارت‌های Rythmicity، Werable، SAA، week of the week و so far ny در فایل‌های فعلی مقاله پیدا نشدند. این فقط وضعیت مقالهٔ فعلی را نشان می‌دهد، نه اصلاح شدن نسخهٔ اصلی proposal.
- نمودار Figure 2.1 با محور درصد بیش از ۱۰۰، شماره‌گذاری فصل‌های proposal و تکرار بخش‌های 2.1/2.1.1 را از مقالهٔ ساختاراً متفاوت فعلی نمی‌توان دقیقاً تطبیق داد؛ منبع proposal لازم است.
- نام Yan در مقدمه درست ذکر شده، اما `References.bib` فعلی فقط دو مدخل CoST و GLOBEM دارد و مدخل مقالهٔ Yan ندارد. در نتیجه اصلاح و تکمیل ارجاع موردنظر استاد هنوز تمام نشده است.
- seasonal در مقدمه با cyclic/periodic توضیح داده می‌شود، اما همان متن عبارتی دربارهٔ تغییر seasonally هم دارد. بهتر است صریحاً نوشته شود seasonal در این مدل یعنی مؤلفهٔ تکرارشونده و الزاماً فصل‌های سال نیست.

## ناسازگاری مستندات که بر این قضاوت اثر دارد

- README می‌گوید اجرای GPU و نتایج نهایی pending است، و گزارش اصلی می‌گوید artefactهای canonical محلی نیستند. هنگام این بررسی، خروجی‌های canonical HRD/GLOBEM واقعاً در `results/` موجود بودند. بنابراین وضعیت اجرا از فایل‌های واقعی بررسی شد.
- بخش روش مقاله می‌گوید «no imputed value contributes»، اما ورودی مدل و هدف‌های ریتم از داده‌های completed استفاده می‌کنند. این عبارت با داشتن observation mask به‌تنهایی درست نمی‌شود.
- بخش downstream مقاله از میانگین احتمال پنجره‌ها سخن می‌گوید، در حالی که کد فعال ابتدا embeddingها را در سطح فرد میانگین می‌گیرد و سپس classifier را اجرا می‌کند. برای پاسخ به استاد باید روش واقعی نوشته شود.
- گزارش، تعریف CES-D endpoint را قطعی بیان می‌کند، ولی metadata کش همچنان verification را pending می‌داند. باید وضعیت verification با مدرک منشأ داده هماهنگ شود؛ از این تناقض به‌تنهایی غلط بودن برچسب‌ها نتیجه گرفته نمی‌شود.

## اولویت اقدام

۱. واژگان و ادعاهای observed rhythm در برابر endogenous circadian را در تمام مقاله یکدست کن.
۲. فاز را در loss، readout، تجمیع، baseline، فاصله، scaling و baselineهای classical یکپارچه دوری بررسی کن؛ گذراندن یک آزمون خطای دوری کافی نیست.
۳. تحلیل حساسیت missingness/imputation و missingness-only baseline را اضافه کن؛ نتیجهٔ تغییر cohort را جدا گزارش کن.
۴. منشأ ابزار، cutoff، زمان و واحد endpoint هر مجموعه را تثبیت کن و روایت HRD در proposal/گزارش/cache را همسان کن.
۵. جدول توصیفی گروه‌ها با آمار دوری، fit quality، اندازهٔ اثر و CI بساز؛ سپس موضوع متغیرهای زمینه‌ای و timezone/DST را با دادهٔ موجود یا محدودیت صریح پوشش بده.

این بررسی هیچ مدل، config، نتیجه یا فایل مقاله را تغییر نداده است؛ فقط همین گزارش افزوده شد. آموزش و بازاجرای کامل آزمایش‌ها انجام نشد.
