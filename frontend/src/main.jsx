import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/noto-sans-kr";
import "./style.css";

const labels = {
  new: "신규",
  review: "답변 검토",
  needs_info: "확인 필요",
  escalated: "담당자 이관",
  resolved: "처리 완료",
};
const orderStates = {
  paid: "결제 완료",
  preparing: "상품 준비 중",
  shipped: "배송 중",
  delivered: "배송 완료",
  cancelled: "주문 취소",
};
const time = (value) =>
  new Date(value).toLocaleString("ko-KR", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
const number = (value) => Number(value).toLocaleString("ko-KR");

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    throw new Error(
      "서버에 연결하지 못했습니다. 실행 상태를 확인하고 다시 시도해 주세요.",
    );
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "입력 내용을 확인하고 다시 시도해 주세요.",
    );
  return data;
}
const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body ?? {}) });

function Icon({ name }) {
  const paths = {
    search: (
      <>
        <circle cx="9" cy="9" r="6" />
        <path d="m14 14 5 5" />
      </>
    ),
    plus: <path d="M10 4v12M4 10h12" />,
    check: <path d="m4 10 4 4L17 5" />,
    file: (
      <>
        <path d="M5 2h7l4 4v12H5zM12 2v5h4M8 11h5M8 14h5" />
      </>
    ),
    refresh: (
      <>
        <path d="M16 7a7 7 0 1 0 1 6M16 3v5h-5" />
      </>
    ),
    person: (
      <>
        <circle cx="8" cy="6" r="3" />
        <path d="M2 18v-2a6 6 0 0 1 12 0v2M17 6v6M14 9h6" />
      </>
    ),
  };
  return (
    <svg
      className="icon"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
function Badge({ status, automatic }) {
  const pending = status === "new" && {
    queued: "초안 대기",
    running: "AI 작성 중",
    failed: "초안 실패",
  }[automatic?.state];
  return <span className={`badge ${automatic?.state === "failed" && status === "new" ? "needs_info" : status}`}>{pending || labels[status] || status}</span>;
}
function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

function App() {
  const [view, setView] = useState("inbox");
  const [status, setStatus] = useState(null);
  const [tickets, setTickets] = useState([]);
  const [customers, setCustomers] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [ticket, setTicket] = useState(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [answer, setAnswer] = useState("");
  const [edited, setEdited] = useState(false);
  const [handoff, setHandoff] = useState(false);
  const [note, setNote] = useState("");
  const [showRegenerate, setShowRegenerate] = useState(false);
  const [guidance, setGuidance] = useState("");
  const displayedTicket = useRef(null);
  const [pollError, setPollError] = useState("");

  async function refresh() {
    const [s, t, c, d] = await Promise.all([
      api("/status"),
      api("/tickets"),
      api("/customers"),
      api("/documents"),
    ]);
    setStatus(s);
    setTickets(t.tickets);
    setCustomers(c.customers);
    setDocuments(d.documents);
    setSelected(
      (current) =>
        current ??
        t.tickets[0]?.id ??
        null,
    );
  }
  async function run(label, task) {
    setBusy(label);
    setError("");
    setNotice("");
    try {
      await task();
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy("");
    }
  }
  useEffect(() => {
    run("문의함을 불러오고 있습니다.", refresh);
  }, []);
  useEffect(() => {
    if (!selected) return;
    const abort = new AbortController();
    setTicket(null);
    api(`/tickets/${selected}`, { signal: abort.signal })
      .then((result) => {
        if (!abort.signal.aborted) setTicket((current) =>
          current?.id === result.id && current.revision > result.revision ? current : result,
        );
      })
      .catch((failure) => {
        if (!abort.signal.aborted) setError(failure.message);
      });
    return () => abort.abort();
  }, [selected]);
  useEffect(() => {
    const initialCompletion =
      ticket?.id === displayedTicket.current?.id &&
      !displayedTicket.current?.draft && !!ticket?.draft;
    displayedTicket.current = ticket;
    setAnswer(ticket?.final_answer || ticket?.draft?.answer || "");
    if (initialCompletion) return;
    setEdited(false);
    setHandoff(false);
    setNote("");
    setShowRegenerate(false);
    setGuidance("");
  }, [ticket?.id, ticket?.revision]);
  useEffect(() => {
    if (!status || busy) return;
    const abort = new AbortController();
    let timer;
    async function poll() {
      try {
        const [nextStatus, nextTickets, detail] = await Promise.all([
          api("/status", { signal: abort.signal }),
          api("/tickets", { signal: abort.signal }),
          view === "inbox" && selected
            ? api(`/tickets/${selected}`, { signal: abort.signal })
            : null,
        ]);
        if (abort.signal.aborted) return;
        setStatus(nextStatus);
        setTickets(nextTickets.tickets);
        setPollError("");
        if (detail) setTicket((current) => {
          if (!current || current.id !== detail.id || detail.revision < current.revision) return current;
          // Polling may fill the first draft, never replace an answer being edited.
          if (!current.draft && !["resolved", "escalated"].includes(current.status)) return detail;
          return current;
        });
      } catch (failure) {
        if (!abort.signal.aborted) setPollError(failure.message);
      } finally {
        if (!abort.signal.aborted) timer = window.setTimeout(poll, 1500);
      }
    }
    timer = window.setTimeout(poll, 1500);
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [selected, view, busy, !!status]);
  useEffect(() => {
    const row = document.querySelector(".ticket-row.selected");
    const list = document.querySelector(".ticket-list");
    if (row && list) list.scrollTop = row.offsetTop - list.offsetTop;
  }, [selected, view]);
  useEffect(() => {
    function prevent(event) {
      if (edited) {
        event.preventDefault();
        event.returnValue = "";
      }
    }
    window.addEventListener("beforeunload", prevent);
    return () => window.removeEventListener("beforeunload", prevent);
  }, [edited]);
  function canLeave() {
    return (
      !edited ||
      window.confirm(
        "아직 기록하지 않은 수정 내용이 있습니다. 수정 내용을 버리고 이동할까요?",
      )
    );
  }
  function navigate(next) {
    if (busy) return;
    if (canLeave()) {
      setEdited(false);
      setAnswer(ticket?.final_answer || ticket?.draft?.answer || "");
      setGuidance("");
      setShowRegenerate(false);
      setView(next);
      setNotice("");
      setError("");
    }
  }
  function select(id) {
    if (id !== selected && canLeave()) {
      setEdited(false);
      setSelected(id);
      setError("");
      setNotice("");
    }
  }
  async function draft() {
    if (!ticket.draft) return;
    await run(
      "입력한 방향과 현재 답변을 참고해 다시 작성하고 있습니다.",
      async () => {
        const result = await post(`/tickets/${ticket.id}/draft`, {
          revision: ticket.revision,
          guidance,
          current_answer: answer,
        });
        setTicket(result);
        await refresh();
        setNotice(
          "재생성 결과로 교체했습니다. 답변과 원문 근거를 검토해 주세요.",
        );
      },
    );
  }
  async function retryDraft() {
    await run("실패한 초안을 다시 작성하도록 요청하고 있습니다.", async () => {
      setTicket(await post(`/tickets/${ticket.id}/retry-draft`, { revision: ticket.revision }));
      await refresh();
    });
  }
  async function resolve(action) {
    await run("처리 결과를 기록하고 있습니다.", async () => {
      const result = await post(`/tickets/${ticket.id}/resolve`, {
        revision: ticket.revision,
        action,
        answer: action === "approve" ? answer : note,
      });
      setTicket(result);
      await refresh();
      setNotice(
        action === "approve"
          ? "승인 내용을 로컬에 기록했습니다. 고객에게 발송하지 않았습니다."
          : "담당자 이관 내용을 로컬에 기록했습니다. 외부 알림은 보내지 않았습니다.",
      );
    });
  }
  const visible = tickets.filter(
    (item) =>
      (filter === "all" || item.status === filter) &&
      `${item.subject} ${item.body} ${customers.find((c) => c.id === item.customer_id)?.name ?? ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const processed = ["resolved", "escalated"].includes(ticket?.status);
  const autoRunning = (status?.auto_draft?.running ?? 0) > 0;
  const autoState = ticket?.auto_draft?.state;
  const stale =
    ticket?.draft &&
    (!status?.indexed ||
      ticket.draft.index_fingerprint !== status.index_fingerprint);
  const canApprove =
    ticket?.draft?.action === "answer" &&
    ticket.draft.citations.length > 0 &&
    !stale &&
    !processed &&
    answer.trim();

  return (
    <>
      <a className="skip-link" href="#main">
        본문으로 이동
      </a>
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(event) => {
            event.preventDefault();
            navigate("inbox");
          }}
        >
          문의데스크
        </a>
        <nav aria-label="주 메뉴">
          {[
            ["inbox", "문의함"],
            ["knowledge", "FAQ 자료"],
            ["settings", "설정"],
          ].map(([id, label]) => (
            <button
              key={id}
              className={view === id ? "active" : ""}
              aria-current={view === id ? "page" : undefined}
              disabled={!!busy}
              onClick={() => navigate(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <label className="search">
          <Icon name="search" />
          <input
            aria-label="문의 검색"
            placeholder="문의 내용, 고객명으로 검색"
            value={query}
            disabled={!!busy}
            onChange={(event) => {
              if (view !== "inbox") {
                if (!canLeave()) return;
                setEdited(false);
                setView("inbox");
              }
              setQuery(event.target.value);
            }}
          />
        </label>
        <p className="demo-label">의류 쇼핑몰 데모 · 외부 발송 없음</p>
      </header>
      <div className="messages" aria-live="polite">
        {busy && <p className="message loading">{busy}</p>}
        {notice && <p className="message success">{notice}</p>}
        {pollError && <p className="message error">상태 갱신을 기다리고 있습니다. {pollError} 자동으로 다시 연결합니다.</p>}
      </div>
      {error && (
        <div className="message error" role="alert">
          {error}
          <button
            onClick={() =>
              run("다시 불러오고 있습니다.", async () => {
                await refresh();
                if (selected) setTicket(await api(`/tickets/${selected}`));
              })
            }
            disabled={!!busy}
          >
            다시 불러오기
          </button>
        </div>
      )}
      {!status ? (
        <main id="main" className="initial">
          <h1>문의함 연결 중</h1>
          <p>의류 쇼핑몰 FAQ와 문의를 불러옵니다.</p>
        </main>
      ) : view === "settings" ? (
        <Settings
          status={status}
          busy={busy}
          run={run}
          refresh={refresh}
          setNotice={setNotice}
          onKnowledge={() => navigate("knowledge")}
        />
      ) : view === "knowledge" ? (
        <Knowledge
          documents={documents}
          status={status}
          busy={busy}
          run={run}
          refresh={refresh}
          setNotice={setNotice}
          markEdited={setEdited}
        />
      ) : view === "new" ? (
        <NewTicket
          customers={customers}
          busy={busy}
          markEdited={setEdited}
          onCancel={() => navigate("inbox")}
          onSubmit={(values) =>
            run("문의를 접수하고 있습니다.", async () => {
              const result = await post("/tickets", values);
              await refresh();
              setEdited(false);
              setSelected(result.id);
              setTicket(result);
              setView("inbox");
              setNotice(
                "문의를 접수했습니다. AI가 순서대로 1차 초안을 작성합니다.",
              );
            })
          }
        />
      ) : (
        <main id="main" className="desk-layout" aria-busy={!!busy}>
          <aside className="queue" aria-label="문의 목록">
            <div className="queue-heading">
              <h2>
                문의 목록 <span>({visible.length})</span>
              </h2>
              <button
                className="small"
                onClick={() => navigate("new")}
                disabled={!!busy}
              >
                <Icon name="plus" />
                문의 접수
              </button>
            </div>
            <p className="queue-automation" role="status">
              {status.auto_draft?.blocked_reason ||
                (autoRunning || status.auto_draft?.queued || status.auto_draft?.failed
                  ? `AI 작성 중 ${status.auto_draft?.running ?? 0}건 · 대기 ${status.auto_draft?.queued ?? 0}건${status.auto_draft?.failed ? ` · 실패 ${status.auto_draft.failed}건` : ""}`
                  : "접수된 문의는 AI가 먼저 초안을 작성합니다.")}
            </p>
            <label className="filter-label">
              상태
              <select
                aria-label="문의 상태 필터"
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                <option value="all">전체 상태</option>
                {Object.entries(labels).map(([id, label]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <div className="ticket-list">
              {visible.map((item) => (
                <button
                  key={item.id}
                  className={`ticket-row ${selected === item.id ? "selected" : ""}`}
                  aria-pressed={selected === item.id}
                  onClick={() => select(item.id)}
                  disabled={!!busy}
                >
                  <span className="ticket-meta">
                    <Badge status={item.status} automatic={item.auto_draft} />
                    <time dateTime={item.created_at}>
                      {time(item.created_at)}
                    </time>
                  </span>
                  <strong>{item.subject}</strong>
                  <span className="muted">
                    {customers.find((c) => c.id === item.customer_id)?.name} ·{" "}
                    {customers.find((c) => c.id === item.customer_id)?.order_number}
                  </span>
                  <span className="excerpt">{item.body}</span>
                </button>
              ))}
              {!visible.length && (
                <p className="empty">검색 조건에 맞는 문의가 없습니다.</p>
              )}
            </div>
          </aside>
          <section className="workspace" aria-label="선택한 문의">
            {!ticket ? (
              <p className="empty">
                {selected
                  ? "문의 내용을 불러오고 있습니다."
                  : "문의 접수 버튼으로 시작해 주세요."}
              </p>
            ) : (
              <>
                <div className="subject">
                  <h1>{ticket.subject}</h1>
                  <Badge status={ticket.status} automatic={ticket.auto_draft} />
                </div>
                <p className="ticket-id">
                  문의 #{ticket.id.slice(0, 8)} · {time(ticket.created_at)}
                </p>
                <section className="customer-panel">
                  <h2>고객·주문 정보</h2>
                  <dl className="customer-facts">
                    <div>
                      <dt>고객 / 주문번호</dt>
                      <dd>{ticket.customer.name}<br /><span className="muted">{ticket.customer.order_number}</span></dd>
                    </div>
                    <div>
                      <dt>주문 상태</dt>
                      <dd>{orderStates[ticket.customer.order_status] ?? "확인 필요"}</dd>
                    </div>
                    <div>
                      <dt>주문 시점</dt>
                      <dd>{ticket.customer.days_since_order === 0 ? "오늘" : `${ticket.customer.days_since_order}일 전`}</dd>
                    </div>
                    <div>
                      <dt>수령 시점</dt>
                      <dd>{ticket.customer.days_since_delivery == null ? "수령 기록 없음" : ticket.customer.days_since_delivery === 0 ? "오늘" : `${ticket.customer.days_since_delivery}일 전`}</dd>
                    </div>
                    <div className="product-fact">
                      <dt>주문 상품</dt>
                      <dd>{ticket.customer.product_name}{" · "}<span className="muted">{ticket.customer.option} · {number(ticket.customer.paid_amount_krw)}원</span></dd>
                    </div>
                  </dl>
                </section>
                <section className="inquiry-panel">
                  <h2>고객 문의 내용</h2>
                  <p>{ticket.body}</p>
                </section>
                <section className="answer-panel">
                  <div className="section-heading">
                    <h2>
                      {processed
                        ? ticket.status === "resolved"
                          ? "승인한 답변"
                          : "이관 기록"
                        : "AI 답변 초안"}
                    </h2>
                    {ticket.draft && !processed && (
                      <button
                        className="small"
                        disabled={!!busy || autoRunning || !status.configured || !status.indexed}
                        aria-expanded={showRegenerate}
                        aria-controls="regenerate-form"
                        onClick={() => setShowRegenerate(!showRegenerate)}
                      >
                        <Icon name="refresh" />
                        재생성
                      </button>
                    )}
                  </div>
                  {!ticket.draft && !processed && (
                    <p className={`guidance ${autoState === "failed" ? "caution" : ""}`} role="status">
                      {autoState === "failed" ? (
                        <>초안을 작성하지 못했습니다. {ticket.auto_draft.error} 원인을 확인한 뒤 재시도해 주세요.</>
                      ) : !status.configured ? (
                        <>
                          먼저{" "}
                          <button
                            className="text-button"
                            onClick={() => navigate("settings")}
                          >
                            설정에서 Gemini 키를 입력
                          </button>
                          해 주세요. 준비가 끝나면 대기 중인 문의의 초안을 자동으로 작성합니다.
                        </>
                      ) : !status.indexed ? (
                        <>
                          FAQ 자료에서{" "}
                          <button
                            className="text-button"
                            onClick={() => navigate("knowledge")}
                          >
                            검색 색인을 만들어 주세요.
                          </button>
                          {" "}완료되면 대기 중인 문의를 자동으로 처리합니다.
                        </>
                      ) : (
                        autoState === "running"
                          ? "AI가 주문 정보와 FAQ를 확인해 1차 초안을 작성하고 있습니다. 완료되면 이곳에 표시됩니다."
                          : "AI 초안 작성 대기 중입니다. 앞선 문의가 끝나면 자동으로 작성합니다."
                      )}
                    </p>
                  )}
                  {ticket.draft && (
                    <p
                      className={`guidance ${ticket.draft.action !== "answer" ? "caution" : ""}`}
                    >
                      <strong>
                        {ticket.draft.category} ·{" "}
                        {
                          {
                            answer: "상담원 검토 필요",
                            clarify: "추가 정보 확인",
                            escalate: "담당자 검토 필요",
                          }[ticket.draft.action]
                        }
                      </strong>
                      <br />
                      {ticket.draft.reason}
                    </p>
                  )}
                  {ticket.draft && !processed && autoRunning && (
                    <p className="field-note">다른 문의의 1차 초안을 작성 중입니다. 첨삭과 승인은 가능하며, 재생성은 자동 작성이 끝난 뒤 사용할 수 있습니다.</p>
                  )}
                  {stale && !processed && (
                    <p className="guidance caution">
                      정책이 변경되었습니다. 지식 색인을 갱신하고 초안을 다시
                      작성해야 승인할 수 있습니다.
                    </p>
                  )}
                  <label className="answer-label">
                    <span className="sr-only">
                      {processed ? "기록한 내용" : "답변 내용"}
                    </span>
                    <textarea
                      aria-label={processed ? "기록한 내용" : "답변 내용"}
                      placeholder="AI가 작성한 1차 초안이 이곳에 표시됩니다. 완료 후 검토하고 수정하세요."
                      maxLength={6000}
                      value={answer}
                      readOnly={
                        processed ||
                        !ticket.draft ||
                        ticket.draft.action !== "answer"
                      }
                      disabled={!!busy}
                      onChange={(event) => {
                        setAnswer(event.target.value);
                        setEdited(true);
                      }}
                    />
                    <span className="character-count">
                      {answer.length.toLocaleString()} / 6,000
                    </span>
                  </label>
                  {showRegenerate && !processed && (
                    <form
                      id="regenerate-form"
                      className="regenerate-form"
                      onSubmit={(event) => {
                        event.preventDefault();
                        draft();
                      }}
                    >
                      <div className="field">
                        <label htmlFor="regenerate-guidance">재생성 방향 (선택)</label>
                        <textarea
                          id="regenerate-guidance"
                          autoFocus
                          rows={3}
                          maxLength={1000}
                          placeholder="예: 더 짧게, 사과 표현을 넣어서, 해결 절차부터 설명"
                          value={guidance}
                          disabled={!!busy}
                          aria-describedby="regenerate-note"
                          onChange={(event) => {
                            setGuidance(event.target.value);
                            setEdited(true);
                          }}
                        />
                      </div>
                      <p id="regenerate-note" className="field-note">
                        현재 답변과 입력한 방향을 함께 참고합니다. 재생성이 완료되면
                        수정 중인 내용을 교체합니다. 정책에 없는 사실은 추가할 수 없습니다.
                      </p>
                      <button
                        className="outline"
                        disabled={!!busy || autoRunning || !status.configured || !status.indexed}
                      >
                        <Icon name="refresh" />
                        {busy ? "처리 중…" : "답변 재생성"}
                      </button>
                      <p
                        role="status"
                        className={`regeneration-status field-note ${error ? "failed" : ""}`}
                      >
                        {busy || (error && `${error} 현재 답변과 입력한 방향은 유지됩니다.`)}
                      </p>
                    </form>
                  )}
                  <p className="field-note">
                    승인 시 이 PC에만 기록됩니다. 고객 발송이나 주문 취소·환불은 실행하지 않습니다.
                  </p>
                  {!processed && (
                    <div className="actions">
                      <button
                        disabled={!!busy}
                        onClick={() => setHandoff(!handoff)}
                        aria-expanded={handoff}
                      >
                        <Icon name="person" />
                        담당자 이관
                      </button>
                      {!ticket.draft && autoState === "failed" && <button
                        className="outline"
                        disabled={
                          !!busy || autoRunning || !status.configured || !status.indexed
                        }
                        onClick={retryDraft}
                      >
                        <Icon name="refresh" />
                        초안 재시도
                      </button>}
                      <button
                        className="primary"
                        disabled={!!busy || !canApprove}
                        onClick={() => resolve("approve")}
                      >
                        <Icon name="check" />
                        승인하고 기록
                      </button>
                    </div>
                  )}
                  {handoff && (
                    <form
                      className="handoff"
                      onSubmit={(event) => {
                        event.preventDefault();
                        resolve("escalate");
                      }}
                    >
                      <Field label="이관 사유">
                        <textarea
                          autoFocus
                          required
                          maxLength={6000}
                          value={note}
                          onChange={(event) => {
                            setNote(event.target.value);
                            setEdited(true);
                          }}
                          placeholder="추가로 확인할 정보와 담당자의 조치가 필요한 이유"
                        />
                      </Field>
                      <button
                        className="primary"
                        disabled={!!busy || !note.trim()}
                      >
                        이관 내용 기록
                      </button>
                    </form>
                  )}
                </section>
                <details className="history">
                  <summary>처리 이력 ({ticket.history.length})</summary>
                  <ol>
                    {ticket.history.map((item, index) => (
                      <li key={index}>
                        <time>{time(item.at)}</time>
                        <p>{item.detail}</p>
                      </li>
                    ))}
                  </ol>
                </details>
              </>
            )}
          </section>
          <Evidence
            draft={ticket?.draft}
            documents={documents}
            onKnowledge={() => navigate("knowledge")}
          />
        </main>
      )}
    </>
  );
}

function Evidence({ draft, documents, onKnowledge }) {
  const [query, setQuery] = useState("");
  const evidence = draft?.evidence ?? [];
  const citations = draft?.citations ?? [];
  const preview = documents.filter((doc) =>
    `${doc.title} ${doc.body}`.includes(query),
  );
  return (
    <aside className="evidence" aria-label="FAQ 및 답변 근거">
      <h2>FAQ 및 답변 근거</h2>
      {draft ? (
        <>
          <p className="field-note">
            검색된 조각 {evidence.length}개 · 인용 {citations.length}개<br />
            검색 순위는 답변의 정확도 점수가 아닙니다.
          </p>
          {!evidence.length && <p className="empty">검색된 근거가 없습니다.</p>}
          {evidence.map((item, index) => {
            const quotes = citations.filter(
              (c) => c.chunk_id === item.chunk_id,
            );
            return (
              <details
                className="source"
                key={item.chunk_id}
                open={quotes.length > 0 || undefined}
              >
                <summary>
                  <span>
                    <span className="source-marker">
                      {quotes.length
                        ? "답변에 인용됨"
                        : `검색 결과 ${index + 1}`}
                    </span>
                    <strong>{item.title}</strong>
                  </span>
                  <span className="version">v{item.version}</span>
                </summary>
                {quotes.map((c, i) => (
                  <blockquote key={i}>{c.quote}</blockquote>
                ))}
                <p className="source-text">{item.text}</p>
                <p className="field-note">
                  출처: 등록된 FAQ · {item.topic}
                </p>
              </details>
            );
          })}
        </>
      ) : (
        <>
          <label className="search policy-search">
            <Icon name="search" />
            <input
              aria-label="FAQ 검색"
              placeholder="FAQ 찾아보기"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </label>
          <p className="guidance">
            아직 이 문의의 검색 근거가 없습니다. 자동 초안 작성이 끝나면 인용한 FAQ와
            원문이 여기에 나타납니다.
          </p>
          <h3>자주 묻는 질문</h3>
          <div className="policy-links">
            {preview.slice(0, 5).map((doc) => (
              <details key={doc.id}>
                <summary>
                  <Icon name="file" />
                  {doc.title}
                </summary>
                <p className="source-text">{doc.body}</p>
              </details>
            ))}
          </div>
          {!preview.length && <p>일치하는 자료가 없습니다.</p>}
          <button className="text-button" onClick={onKnowledge}>
            FAQ {documents.length}개 전체 보기
          </button>
        </>
      )}
    </aside>
  );
}

function Settings({ status, busy, run, refresh, setNotice, onKnowledge }) {
  const [key, setKey] = useState("");
  return (
    <main id="main" className="page settings">
      <h1>설정</h1>
      <p>Gemini 연결 정보를 이 PC에 설정합니다.</p>
      <section>
        <h2>Gemini API 키</h2>
        <p className="guidance">
          {status.configured
            ? "키가 설정되어 있습니다. 연결 성공 여부는 색인을 만들 때 확인됩니다."
            : "아직 키를 설정하지 않았습니다."}
        </p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            run("키를 저장하고 있습니다.", async () => {
              await api("/settings/key", {
                method: "PUT",
                body: JSON.stringify({ api_key: key }),
              });
              setKey("");
              await refresh();
              setNotice(
                "키를 저장했습니다. 최신 검색 색인이 있으면 대기 문의를 자동으로 처리합니다.",
              );
            });
          }}
        >
          <Field
            label="새 API 키"
            hint="서버의 data/clothing/gemini-key에 평문으로 저장됩니다. 이 PC의 파일에 접근할 수 있는 사람은 읽을 수 있으므로 data 폴더를 공유하지 마세요."
          >
            <input
              type="password"
              autoComplete="off"
              spellCheck="false"
              required
              minLength={20}
              maxLength={200}
              value={key}
              onChange={(event) => setKey(event.target.value)}
              placeholder="Gemini 키 전체를 붙여 넣으세요"
            />
          </Field>
          <button className="primary" disabled={!!busy || !key.trim()}>
            키 저장
          </button>
        </form>
      </section>
      <section>
        <h2>전송되는 정보</h2>
        <p>
          색인 생성 시 FAQ가, 답변 작성 시 문의·주문 정보·검색된 FAQ가
          Google Gemini로 전송됩니다. 일반적인 이메일과 국내 전화번호는
          치환하지만 모든 개인정보를 걸러내지는 못합니다. 데모 데이터만 사용해
          주세요.
        </p>
        <p>
          API 호출에는 사용량에 따른 비용이 발생할 수 있습니다. 키와 최신 검색 색인이 준비되면
          대기 중인 문의를 자동으로 분석하며, 이후 접수된 문의도 자동으로 처리합니다.
        </p>
        <dl className="settings-facts">
          <div>
            <dt>답변 모델</dt>
            <dd>{status.model}</dd>
          </div>
          <div>
            <dt>임베딩 모델</dt>
            <dd>{status.embedding_model}</dd>
          </div>
          <div>
            <dt>지식 색인</dt>
            <dd>
              {status.indexed
                ? `${status.chunk_count}개 조각 · 최신`
                : "생성 또는 갱신 필요"}
            </dd>
          </div>
        </dl>
        <button onClick={onKnowledge} disabled={!!busy}>
          FAQ 자료로 이동
        </button>
      </section>
    </main>
  );
}

function Knowledge({
  documents,
  status,
  busy,
  run,
  refresh,
  setNotice,
  markEdited,
}) {
  const [selected, setSelected] = useState(null);
  const [form, setForm] = useState(null);
  const [dirty, setDirty] = useState(false);
  function choose(doc) {
    if (dirty && !window.confirm("저장하지 않은 문서 수정을 버릴까요?")) return;
    setSelected(doc?.id ?? "new");
    setForm(doc ?? { title: "", topic: "", body: "" });
    setDirty(false);
    markEdited(false);
  }
  useEffect(() => {
    const prevent = (event) => {
      if (dirty) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", prevent);
    return () => window.removeEventListener("beforeunload", prevent);
  }, [dirty]);
  function field(name, value) {
    setForm({ ...form, [name]: value });
    setDirty(true);
    markEdited(true);
  }
  return (
    <main id="main" className="page knowledge">
      <div className="page-heading">
        <div>
          <h1>FAQ 자료</h1>
          <p>가상의 의류 쇼핑몰 안내입니다. 실제 매장 정책이나 법률 안내로 사용하지 마세요.</p>
        </div>
        <button
          className="primary"
          disabled={!!busy || !!status.auto_draft?.running || !status.configured}
          onClick={() =>
            run(
              "FAQ를 임베딩하고 검색 색인을 저장하고 있습니다.",
              async () => {
                await post("/index");
                await refresh();
                setNotice(
                  "검색 색인을 갱신했습니다. 대기 중인 문의의 초안을 자동으로 작성합니다.",
                );
              },
            )
          }
        >
          검색 색인 {status.indexed ? "다시 만들기" : "만들기"}
        </button>
      </div>
      <p className={`guidance ${!status.indexed ? "caution" : ""}`}>
        문서 {documents.length}개 ·{" "}
        {status.indexed
          ? `${status.chunk_count}개 조각이 최신 상태로 색인되어 있습니다.`
          : "색인이 없거나 문서가 변경되었습니다. 색인을 만들어야 AI 초안을 작성할 수 있습니다."}
        <br />
        색인 생성은 FAQ를 Gemini로 전송하며 API 사용량이 발생합니다. 완료되면 대기 문의의 초안 작성이 자동으로 이어집니다.{" "}
        {!!status.auto_draft?.running && "현재 초안을 작성 중입니다. 완료 후 색인을 갱신할 수 있습니다. "}
        {!status.configured && "먼저 설정에서 키를 입력해 주세요."}
      </p>
      <div className="knowledge-layout">
        <aside>
          <div className="section-heading">
            <h2>자주 묻는 질문</h2>
            <button
              className="small"
              onClick={() => choose(null)}
              disabled={!!busy}
            >
              <Icon name="plus" />
              문서 추가
            </button>
          </div>
          {documents.map((doc) => (
            <button
              key={doc.id}
              className={`doc-row ${selected === doc.id ? "selected" : ""}`}
              onClick={() => choose(doc)}
              disabled={!!busy}
            >
              <strong>{doc.title}</strong>
              <span>
                {doc.topic} · v{doc.version}
              </span>
            </button>
          ))}
        </aside>
        {form ? (
          <form
            className="document-editor"
            onSubmit={(event) => {
              event.preventDefault();
              run("FAQ를 저장하고 있습니다.", async () => {
                const values = {
                  title: form.title,
                  topic: form.topic,
                  body: form.body,
                };
                const result =
                  selected === "new"
                    ? await post("/documents", values)
                    : await api(`/documents/${selected}`, {
                        method: "PUT",
                        body: JSON.stringify({
                          ...values,
                          version: form.version,
                        }),
                      });
                setForm(result);
                setSelected(result.id);
                setDirty(false);
                markEdited(false);
                await refresh();
                setNotice(
                  "문서를 저장했습니다. 검색 색인을 다시 만들어 주세요.",
                );
              });
            }}
          >
            <h2>{selected === "new" ? "FAQ 추가" : "FAQ 편집"}</h2>
            <Field label="문서 제목">
              <input
                required
                maxLength={160}
                value={form.title}
                onChange={(e) => field("title", e.target.value)}
              />
            </Field>
            <Field label="분류">
              <input
                required
                maxLength={30}
                value={form.topic}
                onChange={(e) => field("topic", e.target.value)}
              />
            </Field>
            <Field
              label="FAQ 내용"
              hint="최소 20자, 최대 20,000자. 개인정보와 실제 고객 자료를 넣지 마세요."
            >
              <textarea
                required
                minLength={20}
                maxLength={20000}
                value={form.body}
                onChange={(e) => field("body", e.target.value)}
              />
            </Field>
            <button className="primary" disabled={!!busy || !dirty}>
              문서 저장
            </button>
          </form>
        ) : (
          <div className="empty">
            <h2>원문을 확인하고 수정하세요</h2>
            <p>
              왼쪽에서 FAQ를 선택하세요. 수정한 내용은 새 버전으로 저장되며,
              이전 답변 초안은 다시 검토해야 합니다.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}

function NewTicket({ customers, busy, onCancel, onSubmit, markEdited }) {
  const [customer, setCustomer] = useState(customers[0]?.id ?? "");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  return (
    <main id="main" className="page new-ticket">
      <h1>문의 접수</h1>
      <p>
        가상 주문을 선택하고 문의를 작성하세요. 실제 개인정보는
        입력하지 마세요.
      </p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ customer_id: customer, subject, body });
        }}
      >
        <Field label="문의 주문">
          <select
            value={customer}
            onChange={(e) => setCustomer(e.target.value)}
          >
            {customers.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} · {c.order_number} · {c.product_name} ({c.option})
              </option>
            ))}
          </select>
        </Field>
        <Field label="문의 제목">
          <input
            required
            maxLength={160}
            value={subject}
            onChange={(e) => {
              setSubject(e.target.value);
              markEdited(true);
            }}
          />
        </Field>
        <Field label="문의 내용">
          <textarea
            required
            minLength={5}
            maxLength={5000}
            value={body}
            onChange={(e) => {
              setBody(e.target.value);
              markEdited(true);
            }}
          />
        </Field>
        <div className="form-actions">
          <button type="button" onClick={onCancel} disabled={!!busy}>
            취소
          </button>
          <button className="primary" disabled={!!busy}>
            문의 접수
          </button>
        </div>
      </form>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
