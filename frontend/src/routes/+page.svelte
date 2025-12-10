<script>
	import { onMount } from 'svelte';
	import Chat from '$lib/components/Chat.svelte';
	import Sidebar from '$lib/components/Sidebar.svelte';

	let showSidebar = true;

	function toggleSidebar() {
		showSidebar = !showSidebar;
	}
</script>

<div class="app">
	<header>
		<h1>RPG Assistant</h1>
		<div class="header-actions">
			<button on:click={toggleSidebar} class="toggle-sidebar">
				{showSidebar ? '◀' : '▶'}
			</button>
		</div>
	</header>

	<div class="main-container">
		{#if showSidebar}
			<aside class="sidebar">
				<Sidebar />
			</aside>
		{/if}

		<main class="chat-area">
			<Chat />
		</main>
	</div>
</div>

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
		background-color: #1a1a1a;
		color: #e0e0e0;
	}

	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
	}

	header {
		background-color: #2a2a2a;
		padding: 1rem 2rem;
		display: flex;
		justify-content: space-between;
		align-items: center;
		border-bottom: 1px solid #3a3a3a;
	}

	header h1 {
		margin: 0;
		font-size: 1.5rem;
		color: #4a9eff;
	}

	.header-actions {
		display: flex;
		gap: 1rem;
		align-items: center;
	}

	.toggle-sidebar {
		background: #3a3a3a;
		border: none;
		color: #e0e0e0;
		padding: 0.5rem 1rem;
		cursor: pointer;
		border-radius: 4px;
		font-size: 1rem;
	}

	.toggle-sidebar:hover {
		background: #4a4a4a;
	}

	.main-container {
		display: flex;
		flex: 1;
		overflow: hidden;
	}

	.sidebar {
		width: 300px;
		background-color: #2a2a2a;
		border-right: 1px solid #3a3a3a;
		overflow-y: auto;
	}

	.chat-area {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}
</style>
