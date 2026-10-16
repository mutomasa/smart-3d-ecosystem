import { App } from "./app/App";

const container = document.getElementById("app")!;
const hud = document.getElementById("hud")!;
const app = new App(container, hud);
app.start();
(window as unknown as { __app: App }).__app = app;
