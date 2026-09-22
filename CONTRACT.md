# Backend / UI contract

Local app: 127.0.0.1:8770. API JSON uses snake_case. Error: `{detail: string}`. UI never stores API key. All dataset entries are synthetic demo data.

Active domain: clothing-shopping FAQ. Default runtime data: `data/clothing/`; legacy `data/desk.sqlite3` and `data/vectors/` are preserved, not loaded. Each profile represents one selected single-item order.

- GET /api/status → `{configured, indexed, document_count, chunk_count, ticket_count, model, embedding_model, index_fingerprint, auto_draft:{enabled,queued,running,failed,blocked_reason}}`. Status reads do not wait for model generation.
- PUT /api/settings/key `{api_key}` → `{saved:true}`; not authentication verification. Wakes pending first drafts when the index is ready.
- GET /api/customers → `{customers:[{id,name,order_number,product_name,option,order_status,paid_amount_krw,days_since_order,days_since_delivery}]}`; order_status is paid/preparing/shipped/delivered/cancelled; paid_amount_krw and days_since_order are nonnegative integers; days_since_delivery is a nonnegative integer or null. Order elapsed days are fixed fictional snapshot values, not live timestamps.
- GET /api/documents → `{documents:[{id,title,topic,version,body}]}`
- POST /api/documents `{title,topic,body}` → document; duplicate content rejected.
- PUT /api/documents/{id} `{title,topic,body,version}` → updated doc; conflicts 409. Index becomes stale.
- POST /api/index → `{document_count,chunk_count,index_fingerprint}`; actual embedding calls; bounded local operation.
- GET /api/tickets → `{tickets:[ticket]}` newest first.
- POST /api/tickets `{customer_id,subject,body}` → ticket with queued automatic first draft. Unrelated content is treated as untrusted user input.
- GET /api/tickets/{id} → ticket including customer, draft and history.
- POST /api/tickets/{id}/draft `{revision,guidance?:string,current_answer?:string}` → ticket; manual regeneration of an existing draft only in the normal automatic server (missing draft returns 409). Optional fields default to empty strings (guidance max 1,000 characters; current_answer max 6,000). The original inquiry drives retrieval; separate reviewer context guides wording without overriding policy/customer facts. Successful generation replaces the current draft; failure makes no write. Regeneration history includes the requested guidance. Current index required; AI can return answer/clarify/escalate. No dummy answer fallback. Internal deterministic tests may disable the automatic worker and use this endpoint for initial generation.
- POST /api/tickets/{id}/retry-draft `{revision}` → ticket; explicitly requeues only an unprocessed, failed first draft with no existing draft. Duplicate retries and stale revisions return 409.
- POST /api/tickets/{id}/resolve `{revision,action:'approve'|'escalate',answer:string}` → ticket. Approve requires current supported answer draft, citations and current policy fingerprint. Escalate requires a note. Resolved tickets cannot be modified.

Ticket: `{id,customer_id,subject,body,status:'new'|'review'|'needs_info'|'escalated'|'resolved',revision,created_at,updated_at,auto_draft:{state:'queued'|'running'|'completed'|'failed',error:string},draft:null|{category,action,answer,reason,citations:[{chunk_id,quote}],evidence:[{chunk_id,document_id,title,topic,text,version,score}],index_fingerprint},final_answer,history:[{at,event,detail}]}`.

One server lifespan worker processes pending first drafts serially, independently of browser reads. Startup, new inquiry, key save and completed index operations wake it. Missing key or stale index leaves jobs queued. Existing drafts and resolved/escalated tickets are preserved. Model failure is persisted without an automatic retry loop; an interrupted running job is marked failed on startup. Generation commits only while the ticket revision and policy fingerprint still match. Run one server process, not multiple Uvicorn workers. Browser polling retrieves state only and preserves human edits and form input.

실행 데이터와 API 키는 소스·데모 JSON·검사 결과물과 분리합니다. 이 계약은 가상 주문의 상담 초안 범위이며 실제 주문 시스템이나 고객 발송 API를 포함하지 않습니다.
